#pragma once

#include <cmath>
#include "LRI.h"
#include "LRI_Cal_Aux.h"
#include "../global/Array_Operator.h"
#include "../global/Tensor_Multiply.h"
#include "/home/fisherd/deepmodeling/abacus-BSE/source/source_base/timer.h"
#include "/home/fisherd/deepmodeling/abacus-BSE/source/source_base/tool_title.h"
#include "/home/fisherd/deepmodeling/abacus-BSE/source/source_lcao/module_lr/utils/lr_util_print.h"
#include <omp.h>
#include <malloc.h>
#ifdef __MKL_RI
#include <mkl_service.h>
#endif

namespace RI
{
// used for q=k2-k1 comparison in std::map
struct Tk_Comparator {
	bool operator()(const Tk& lhs, const Tk& rhs) const {
		constexpr double epsilon = 1e-6;
		for (int i = 0; i < 3; ++i) {
			if (std::abs(lhs[i] - rhs[i]) > epsilon) {
				return lhs[i] < rhs[i];
			}
		}
		return false;
	}
};

template<class Map, class Key>
const typename Map::mapped_type* find_map(const Map& map, const Key& key)
{
	const auto ptr = map.find(key);
	if (ptr != map.end()) {
		return std::addressof(ptr->second);
	}
	return nullptr;
}

inline void print_a(const std::vector<int>& vec, const std::string name)
{
	std::cout << name << ": ";
	for (auto& v : vec) { std::cout << v << " "; }
	std::cout << std::endl;
}

inline void print_ac(const std::vector<std::pair<int,std::array<int,3>>>& vec, const std::string name)
{
	std::cout << name << ": ";
	for (auto& v : vec) {
		std::cout << v.first << ": (" << v.second[0] <<", "<< v.second[1] <<", "<< v.second[2] <<")"<< std::endl;
	}
	std::cout << std::endl;
}

inline void print_k(const std::vector<Tk>& vec, const std::string name)
{
	std::cout << name << ": ";
	for (auto& v : vec) {
		std::cout << "(" << v[0] <<", "<< v[1] <<", "<< v[2] <<")"<< std::endl;
	}
	std::cout << std::endl;
}

template<typename Tdata, typename Tfac>
inline void FT_Ds(const Tensor<Tdata>& D_in, Tensor<Tdata>& D_out, const Tfac fac)
{
	if (D_out.empty())
	{
		if (1.0==fac)
			D_out = D_in.copy();
		else
			D_out = Tdata(fac) * D_in;
	}
	else
	{
		if (1.0==fac)
			D_out += D_in;
		else
			D_out += Tdata(fac) * D_in;
	}
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
std::map<Tk, std::map<Tk, Tensor<Tdata>>> 
LRI<TA, Tcell, Ndim, Tdata>::cal_cvc_mo_k(
	std::map<std::pair<Tk, Tk>, std::map<TA, RI::Tensor<Tdata>>>& Cs_oo_mo,
	std::map<std::pair<Tk, Tk>, std::map<TA, RI::Tensor<Tdata>>>& Cs_vv_mo,
	std::vector<Tk>& k1_list,
	std::vector<Tk>& k2_list)
// TODO: k_list is for the key of cvc_mo_k, and is planned to be MPI distributed
{
	using namespace Array_Operator;
	#ifdef __MKL_RI
	const std::size_t mkl_threads = mkl_get_max_threads();
	std::cout << "MKL threads max: " << mkl_threads << std::endl;
	mkl_set_num_threads(1);
	#endif

	std::map<Tk, std::map<Tk, Tensor<Tdata>>> cvc_mo_k;
	std::map<Tk, std::map<std::pair<TA,TA>, Tensor<Tdata>>> Vqs;
	std::map<Tk, std::map<std::pair<TA,TA>, Tensor<Tdata>>, Tk_Comparator> Vqs_fuzzy;
	print_k(k1_list, "k1_list");
	print_k(k2_list, "k2_list");

	// check if VR has all atoms I
	const std::map<TA, std::map<TAC, Tensor<Tdata>>>& Vs = this->data_pool.at("Vs_").Ds_ab;

	const std::vector<TA> list_I = Global_Func::map_key_to_vec(Vs);
	print_a(list_I, "list_I");

	std::set<Tk, Tk_Comparator> q_set;
	Tk k_unit{1.0, 1.0, 1.0};
    for (const Tk& k1 : k1_list)
        for (const Tk& k2 : k2_list)
            q_set.insert( (k2 - k1) % k_unit );
	std::vector<Tk> q_list(q_set.begin(), q_set.end());
	print_k(q_list, "q_list");
	q_set.clear();

	// add thread lock for the first Tk key of cvc_mo_k
	std::map<Tk, omp_lock_t> lock_cvc_result_add_map = LRI_Cal_Aux::init_lock_result(cvc_mo_k, k1_list);
	// add thread lock for the Tk key of Vq
	std::map<Tk, omp_lock_t> lock_vq_result_add_map = LRI_Cal_Aux::init_lock_result(Vqs, q_list);
	#pragma omp parallel
	{
		#pragma omp master
		{
			ModuleBase::TITLE("LRI", "cal_cvc_mo_k");
			ModuleBase::timer::tick("LRI", "cal_cvc_mo_FT_Vq");
		}
	
		// 1. FT V_mu_nu <I,<J,R>> to V_mu_nu <q,<I,J>>
		std::map<Tk, std::map<std::pair<TA,TA>, Tensor<Tdata>>> Vqs_thread;
#pragma omp for schedule(static)
		for (Tk q : q_list)
		{
			auto& Vq = Vqs_thread[q];
			for (const auto &V_mu : Vs)
			{
				TA mu = V_mu.first;
				for (auto V_mu_nu : V_mu.second)
				{
					const TA nu = V_mu_nu.first.first;
					const TC R = V_mu_nu.first.second;
					const Tensor<Tdata>& V_mu_nu_R = V_mu_nu.second;
					double arg = 2.0 * M_PI * (q[0] * R[0] + q[1] * R[1] + q[2] * R[2]);
					std::complex<double> fac (cos(arg), sin(arg));
					FT_Ds(V_mu_nu_R, Vq[std::make_pair(mu, nu)], RI::Global_Func::convert<Tdata>(fac));
				}
			}
			LRI_Cal_Aux::add_Ds_omp_try_map(Vqs_thread, Vqs, lock_vq_result_add_map, 1.0);
			// in theory, each q only belongs to one thread, so try lock should always succeed
		}
		LRI_Cal_Aux::add_Ds_omp_wait_map(Vqs_thread, Vqs, lock_vq_result_add_map, 1.0);

		#pragma omp barrier
		#pragma omp master
		{
			LRI_Cal_Aux::destroy_lock_result(lock_vq_result_add_map, Vqs);
			Vqs_fuzzy.insert(std::make_move_iterator(Vqs.begin()), std::make_move_iterator(Vqs.end()));
			Vqs.clear();
			ModuleBase::timer::tick("LRI", "cal_cvc_mo_FT_Vq");
			ModuleBase::timer::tick("LRI", "cal_cvc_mo_k");
		}
		#pragma omp barrier

		// 2. calculate CVC_mo_k
		std::map<Tk, std::map<Tk, Tensor<Tdata>>> cvc_mo_k_thread;
#pragma omp for schedule(static) collapse(2)
		for (Tk k1: k1_list)
		{
			for (Tk k2: k2_list)
			{
				const auto* Cs_mu_oo_ptr = find_map(Cs_oo_mo, std::make_pair(k2, k1));
				if (Cs_mu_oo_ptr == nullptr)
				#pragma omp critical
				{
					std::cout << "Warning: Cs_oo_mo not found for k2 = (" << k2[0] << ", " << k2[1] << ", " << k2[2] << 
						"), k1 = ("	<< k1[0] << ", " << k1[1] << ", " << k1[2] << ")" << std::endl;
				}
				const auto* Cs_nu_vv_ptr = find_map(Cs_vv_mo, std::make_pair(k1, k2));
				if (Cs_nu_vv_ptr == nullptr) 
				#pragma omp critical
				{
					std::cout << "Warning: Cs_vv_mo not found for k1 = (" << k1[0] << ", " << k1[1] << ", " << k1[2] << 
						"), k2 = ("	<< k2[0] << ", " << k2[1] << ", " << k2[2] << ")" << std::endl;
				};
				const auto& C_mu_oo = *Cs_mu_oo_ptr;
				const auto& C_nu_vv = *Cs_nu_vv_ptr;

				Tk q = (k2-k1) % k_unit;
				const auto* Vq_ptr = find_map(Vqs_fuzzy, q);
				if (Vq_ptr == nullptr) 
				#pragma omp critical
				{
					std::cout << "Warning: Vq not found for q = (" << q[0] << ", " << q[1] << ", " << q[2] << ")" << std::endl;
				}
				const auto& Vq = *Vq_ptr;
				for (auto& c1 : C_mu_oo)
				{
					const TA mu = c1.first;
					const Tensor<Tdata>& C_mu_ji = c1.second;
					for (auto& c2 : C_nu_vv)
					{
						const TA nu = c2.first;
						const Tensor<Tdata>& C_nu_ab = c2.second;
						const auto* Vq_mu_nu_ptr = find_map(Vq, std::make_pair(mu, nu));
						if (Vq_mu_nu_ptr == nullptr) 
						#pragma omp critical
						{
							std::cout << "Warning: Vq_mu_nu not found for mu = " << mu << ", nu = " << nu << std::endl;
						};
						const Tensor<Tdata>& Vq_mu_nu = *Vq_mu_nu_ptr;
						// CV_{ji,nu} = C^mu_{ji} V_{mu,nu}
						const Tensor<Tdata> CV_ji_nu = Tensor_Multiply::x1x2y1_ax1x2_ay1(C_mu_ji, Vq_mu_nu);
						// [CVC]_{ia,jb} += [CV]_{ji,nu} C^nu_{ab}
						// (ji,nu) * (nu,ab) = (jiab) -> (jbia)
						LRI_Cal_Aux::add_Ds(
							Tensor_Multiply::x0x1y1y2_x0x1a_ay1y2(CV_ji_nu, C_nu_ab).permute_from({ 0,3,1,2 }),
							cvc_mo_k_thread[k1][k2]);
					}
				}
				LRI_Cal_Aux::add_Ds_omp_try_map(cvc_mo_k_thread, cvc_mo_k, lock_cvc_result_add_map, 1.0);
			} // end for k2
		} // end for k1
		LRI_Cal_Aux::add_Ds_omp_wait_map(cvc_mo_k_thread, cvc_mo_k, lock_cvc_result_add_map, 1.0);
	} // end #pragma omp parallel

	LRI_Cal_Aux::destroy_lock_result(lock_cvc_result_add_map, cvc_mo_k);

	#ifdef __MKL_RI
	mkl_set_num_threads(mkl_threads);
	#endif

	malloc_trim(0);
	ModuleBase::timer::tick("LRI", "cal_cvc_mo_k");
	return cvc_mo_k;
}	// end LRI::cal_cvc_mo_k



template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
std::map<std::array<Tcell, Ndim>, std::map<std::array<Tcell, Ndim>, Tensor<Tdata>>> 
LRI<TA, Tcell, Ndim, Tdata>::cal_cvc_mo_R(
	std::map<TA, std::map<std::pair<TC, TC>, RI::Tensor<Tdata>>>& Cs_oo_mo,
	std::map<TA, std::map<std::pair<TC, TC>, RI::Tensor<Tdata>>>& Cs_vv_mo,
	std::vector<TC>& R_list)
// TODO: R_list is for the first key of cvc_mo_R, and is planned to be MPI distributed
{
	using namespace Array_Operator;

	const Data_Pack_Wrapper<TA,TC,Tdata> data_wrapper(this->data_pool, this->data_ab_name);
	const LRI_Cal_Tools<TA,TC,Tdata> tools(this->period, this->data_pool, this->data_ab_name);

  #ifdef __MKL_RI
	const std::size_t mkl_threads = mkl_get_max_threads();
	mkl_set_num_threads(1);
  #endif

	std::map<TC, std::map<TC, Tensor<Tdata>>> cvc_mo_R;

	// add thread lock for TC key of cvc_mo_R
	std::map<TC, omp_lock_t> lock_cvc_result_add_map = LRI_Cal_Aux::init_lock_result(cvc_mo_R, R_list);
	
	#pragma omp parallel
	{
		std::map<TC, std::map<TC, Tensor<Tdata>>> cvc_mo_R_thread;

		const std::vector<TA>  list_I0 = LRI_Cal_Aux::filter_list_map(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).a01, data_wrapper(Label::ab::a).Ds_ab);
		const std::vector<TAC> list_J0 = LRI_Cal_Aux::filter_list_map(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).b01, data_wrapper(Label::ab::b).Ds_ab);
		const std::vector<TAC> list_K = LRI_Cal_Aux::filter_list_set(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).a2, data_wrapper(Label::ab::a).index_Ds_ab[0]);
		const std::vector<TAC> list_L = LRI_Cal_Aux::filter_list_set(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).b2, data_wrapper(Label::ab::b).index_Ds_ab[0]);
		// filter Cs by the range of Vs
		const std::vector<TA>  list_I = LRI_Cal_Aux::filter_list_map(list_I0, data_wrapper(Label::ab::a0b0).Ds_ab);
		const std::vector<TAC> list_J = LRI_Cal_Aux::filter_list_set(list_J0, data_wrapper(Label::ab::a0b0).index_Ds_ab[0]);

		// ATTENTION: now we need all V_mu_nu <I,<J,R>> in list_I, list_J temporarily
        #pragma omp single nowait
        {
            print_a(list_I0, "list_I0");
            print_ac(list_J0, "list_J0");
            print_ac(list_K, "list_K");
            print_ac(list_L, "list_L");
            print_a(list_I, "list_I");
            print_ac(list_J, "list_J");
        }
#pragma omp for schedule(static) collapse(2) nowait
		for (TA mu : list_I)
		{
			for (TAC nu_mu : list_J)
			{
				const Tensor<Tdata>& V_mu_nu = tools.get_Ds_ab(Label::ab::a0b0, mu, nu_mu);
				if (V_mu_nu.empty()) continue;
				const TA nu = nu_mu.first;
				const TC R_nu_mu = nu_mu.second;

				const auto* Cs_mu_oo_ptr = find_map(Cs_oo_mo, mu);
				if (Cs_mu_oo_ptr == nullptr) continue;
				const auto* Cs_nu_vv_ptr = find_map(Cs_vv_mo, nu);
				if (Cs_nu_vv_ptr == nullptr) continue;
				const auto& C_mu_oo = *Cs_mu_oo_ptr;
				const auto& C_nu_vv = *Cs_nu_vv_ptr;
				for (auto& c1 : C_mu_oo)
				{
					const TC R_j_mu = c1.first.first;
					const TC R_i_mu = c1.first.second;
					const Tensor<Tdata>& C_mu_ji = c1.second;
					// CV_{ji,nu} = C^mu_{ji} V_{mu,nu}
					const Tensor<Tdata> CV_ji_nu = Tensor_Multiply::x1x2y1_ax1x2_ay1(C_mu_ji, V_mu_nu);
					
					for (auto& c2 : C_nu_vv)
					{
						const TC R_a_nu = c2.first.first;
						const TC R_b_nu = c2.first.second;
						const Tensor<Tdata>& C_nu_ab = c2.second;

						const TC R_ai = (R_a_nu + R_nu_mu - R_i_mu) % period;
						const TC R_bj = (R_b_nu + R_nu_mu - R_j_mu) % period;
						// [CVC]_{ia,jb}+=[CV]_{ji,nu} C^nu_{ab}	
						// (ji,nu) * (nu,ab) = (jiab) -> (jbia)
						LRI_Cal_Aux::add_Ds(
							Tensor_Multiply::x0x1y1y2_x0x1a_ay1y2(CV_ji_nu, C_nu_ab).permute_from({ 0,3,1,2 }),
							cvc_mo_R_thread[R_ai][R_bj]);
					}						
				}
				LRI_Cal_Aux::add_Ds_omp_try_map(cvc_mo_R_thread, cvc_mo_R, lock_cvc_result_add_map, 1.0);
			}
		}
		LRI_Cal_Aux::add_Ds_omp_wait_map(cvc_mo_R_thread, cvc_mo_R, lock_cvc_result_add_map, 1.0);
	} // end #pragma omp parallel

	LRI_Cal_Aux::destroy_lock_result(lock_cvc_result_add_map, cvc_mo_R);

  #ifdef __MKL_RI
	mkl_set_num_threads(mkl_threads);
  #endif

	malloc_trim(0);
	return cvc_mo_R;
}	// end LRI::cal_cvc_mo_R

}	// end namespace RI

