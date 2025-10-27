#pragma once

#include "LRI.h"
#include "LRI_Cal_Aux.h"
#include "../global/Array_Operator.h"
#include "../global/Tensor_Multiply.h"
#include "/home/fisherd/deepmodeling/abacus-BSE/source/source_base/timer.h"
#include "/home/fisherd/deepmodeling/abacus-BSE/source/source_base/tool_title.h"
#include <omp.h>
#include <malloc.h>
#ifdef __MKL_RI
#include <mkl_service.h>
#endif

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
std::map<std::array<Tcell, Ndim>, std::map<std::array<Tcell, Ndim>, Tensor<Tdata>>> 
LRI<TA, Tcell, Ndim, Tdata>::cal_cvc_mo(
	std::map<TA, std::map<std::pair<TC, TC>, RI::Tensor<Tdata>>>& Cs_oo_mo,
	std::map<TA, std::map<std::pair<TC, TC>, RI::Tensor<Tdata>>>& Cs_vv_mo,
	std::vector<TC>& R_list)
{
	ModuleBase::TITLE("LRI", "cal_cvc_mo");
	ModuleBase::timer::tick("LRI", "cal_cvc_mo");
	using namespace Array_Operator;

	const Data_Pack_Wrapper<TA,TC,Tdata> data_wrapper(this->data_pool, this->data_ab_name);
	const LRI_Cal_Tools<TA,TC,Tdata> tools(this->period, this->data_pool, this->data_ab_name);

  #ifdef __MKL_RI
	const std::size_t mkl_threads = mkl_get_max_threads();
	mkl_set_num_threads(1);
  #endif

	std::map<TC, std::map<TC, Tensor<Tdata>>> cvc_mo;

	// add thread lock for TC key of cvc_mo
	std::map<TC, omp_lock_t> lock_cvc_result_add_map = LRI_Cal_Aux::init_lock_result(cvc_mo, R_list);
	
	#pragma omp parallel
	{
		std::map<TC, std::map<TC, Tensor<Tdata>>> cvc_mo_thread;

		// for debug
		auto print_a = [](const std::vector<TA>& vec, const std::string name) -> void
			{
				std::cout << name << ": ";
				for (auto& v : vec) { std::cout << v << " "; }
				std::cout << std::endl;
			};
		auto print_ac = [](const std::vector<TAC>& vec, const std::string name) -> void
			{
				std::cout << name << ": ";
				for (auto& v : vec) {
					std::cout << v.first << ": (" << v.second[0] <<", "<< v.second[1] <<", "<< v.second[2] <<")"<< std::endl;
				}
				std::cout << std::endl;
			};
		const std::vector<TA>  list_I0 = LRI_Cal_Aux::filter_list_map(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).a01, data_wrapper(Label::ab::a).Ds_ab);
		const std::vector<TAC> list_J0 = LRI_Cal_Aux::filter_list_map(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).b01, data_wrapper(Label::ab::b).Ds_ab);
		const std::vector<TAC> list_K = LRI_Cal_Aux::filter_list_set(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).a2, data_wrapper(Label::ab::a).index_Ds_ab[0]);
		const std::vector<TAC> list_L = LRI_Cal_Aux::filter_list_set(this->parallel->list_A.at(Label::Aab_Aab::a01b01_a2b2).b2, data_wrapper(Label::ab::b).index_Ds_ab[0]);
		// filter Cs by the range of Vs
		const std::vector<TA>  list_I = LRI_Cal_Aux::filter_list_map(list_I0, data_wrapper(Label::ab::a0b0).Ds_ab);
		const std::vector<TAC> list_J = LRI_Cal_Aux::filter_list_set(list_J0, data_wrapper(Label::ab::a0b0).index_Ds_ab[0]);

		// TODO: now we need all V_mu_nu <I,<J,R>> in list_I, list_J temporarily
        #pragma omp master
        {
            print_a(list_I0, "list_I0");
            print_ac(list_J0, "list_J0");
            print_ac(list_K, "list_K");
            print_ac(list_L, "list_L");
            print_a(list_I, "list_I");
            print_ac(list_J, "list_J");
        }
		auto find_Cs_mo = [](auto& map, const TA& key) -> decltype(map.begin()->second)*
		{
			const auto ptr = map.find(key);
			if (ptr != map.end()) {
				return std::addressof(ptr->second);
			} else {
				return nullptr;
			}
		};
#pragma omp for schedule(static) collapse(2) nowait
		for (TA mu : list_I)
		{
			for (TAC nu_mu : list_J)
			{
				const Tensor<Tdata>& V_mu_nu = tools.get_Ds_ab(Label::ab::a0b0, mu, nu_mu);
				if (V_mu_nu.empty()) continue;
				const TA nu = nu_mu.first;
				const TC R_nu_mu = nu_mu.second;

				auto* Cs_mu_oo_ptr = find_Cs_mo(Cs_oo_mo, mu);
				if (Cs_mu_oo_ptr == nullptr || Cs_mu_oo_ptr->empty()) continue;
				auto* Cs_nu_vv_ptr = find_Cs_mo(Cs_vv_mo, nu);
				if (Cs_nu_vv_ptr == nullptr || Cs_nu_vv_ptr->empty()) continue;
				auto Cs_mu_oo = *Cs_mu_oo_ptr;
				auto Cs_vv_mo = *Cs_nu_vv_ptr;
				for (auto& c1 : Cs_mu_oo)
				{
					const TC R_j_mu = c1.first.first;
					const TC R_i_mu = c1.first.second;
					const Tensor<Tdata>& C_mu_ji = c1.second;
					// CV_{ji,nu} = C^mu_{ji} V_{mu,nu}
					const Tensor<Tdata> CV_ji_nu = Tensor_Multiply::x1x2y1_ax1x2_ay1(C_mu_ji, V_mu_nu);
					
					for (auto& c2 : Cs_vv_mo)
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
							cvc_mo_thread[R_ai][R_bj]);
					}						
				}
				LRI_Cal_Aux::add_Ds_omp_try_map(cvc_mo_thread, cvc_mo, lock_cvc_result_add_map, 1.0);
			}
		}
		LRI_Cal_Aux::add_Ds_omp_wait_map(cvc_mo_thread, cvc_mo, lock_cvc_result_add_map, 1.0);
	} // end #pragma omp parallel

	LRI_Cal_Aux::destroy_lock_result(lock_cvc_result_add_map, cvc_mo);

  #ifdef __MKL_RI
	mkl_set_num_threads(mkl_threads);
  #endif

	ModuleBase::timer::tick("LRI", "cal_cvc_mo");
	malloc_trim(0);
	return cvc_mo;
}	// end LRI::cal_cvc_mo

}	// end namespace RI

