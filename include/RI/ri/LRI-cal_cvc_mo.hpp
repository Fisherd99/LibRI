#pragma once

#include <cmath>
#include "LRI.h"
#include "LRI_Cal_Aux.h"
#include "../global/Array_Operator.h"
#include "../global/Tensor_Multiply.h"
#include <omp.h>
#include <malloc.h>
#ifdef __MKL_RI
#include <mkl_service.h>
#endif

namespace RI
{
// this fuzzy comparator is used for q=k2-k1 comparison in std::map
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

inline void print_k(std::ostream& ofs, const std::vector<Tk>& vec, const std::string name)
{
	ofs << name << ": size = " << vec.size() << std::endl;
	ofs << std::fixed << std::setprecision(4);
	int count = 0;
	for (auto& v : vec)
	{
		ofs << "(" << std::setw(6) << v[0] <<", "<< std::setw(6) << v[1] <<", "<< std::setw(6) << v[2] <<") ";
		count++;
		if (count % 5 == 0) { ofs << std::endl; }
	}
	ofs << std::endl;
	ofs << std::defaultfloat;
}

inline void switch_mo_type(const std::string &type,
						   std::size_t &imo, std::size_t &nmo,
						   std::size_t nocc, std::size_t nvirt)
{
	if (type == "O")
	{
		imo = 0;
		nmo = nocc;
	}
	else if (type == "V")
	{
		imo = nocc;
		nmo = nvirt;
	}
	else
	{
		throw std::runtime_error("Error in Cs_ao_mo_to_Cs_mo: unknown mo type " + type);
	}
}

/// @brief left_conj: C^mu (m1^*,m2)[k1,k2] = c^*(m1,s)[k1] C^mu (s,m2)[k2] + C^*mu (s,m1)[k1] c(m2,s)[k2]
///    if right_conj: C^mu (m1,m2^*)[k1,k2] = c(m1,s)[k1] C^*mu (s,m2)[k2] + C^mu (s,m1)[k1] c^*(m2,s)[k2]
/// s: atom orbital index; m: band index
template<typename TA, typename Tdata>
inline Tensor<Tdata> Cs_ao_mo_to_Cs_mo(
	const std::map<Tk, std::map<TA, Tensor<Tdata>>>& Cs_ao_mo,  // C^mu (s,m)[k]
	const std::map<Tk, std::map<TA, Tensor<Tdata>>>& psi,       // c(m,s)[k]
	const Tk k1, const Tk k2, TA mu, 
	const std::string& m1_type, const std::string& m2_type,
	const std::size_t nocc, const std::size_t nvirt, const bool is_left_conj)
{
	const Tensor<Tdata>& Cs_ao_mo_k1 = Cs_ao_mo.at(k1).at(mu);
	const Tensor<Tdata>& Cs_ao_mo_k2 = Cs_ao_mo.at(k2).at(mu);
	const Tensor<Tdata>& psi_k1 = psi.at(k1).at(mu);
	const Tensor<Tdata>& psi_k2 = psi.at(k2).at(mu);
	const std::size_t nabf = Cs_ao_mo_k1.shape[0];
	const std::size_t nw = Cs_ao_mo_k1.shape[1];
	const std::size_t nband = Cs_ao_mo_k1.shape[2];
	assert(nband == nocc + nvirt);

	std::size_t imo1, nmo1, imo2, nmo2;
	switch_mo_type(m1_type, imo1, nmo1, nocc, nvirt);
	switch_mo_type(m2_type, imo2, nmo2, nocc, nvirt);
	Tensor<Tdata> Cs_mo({ nabf, nmo1, nmo2 });
	if (is_left_conj)
	{
		std::vector<Tdata> psi_k1_conj(nw * nmo1);
		for (std::size_t m1 = 0; m1 < nmo1; ++m1)
			for (std::size_t iw = 0; iw < nw; ++iw)
				psi_k1_conj[m1 * nw + iw] = Global_Func::get_conj(psi_k1(imo1 + m1, iw));
		for (std::size_t iabf = 0; iabf < nabf; ++iabf)
		{
			Tdata *ptr_out = &Cs_mo(iabf, 0, 0);
			int lda = nw;
			int ldb = nband;
			int ldc = nmo2;
			Blas_Interface::gemm(
				'N', 'N',
				nmo1, nmo2, nw,
				Tdata(1.0), psi_k1_conj.data(), lda, &Cs_ao_mo_k2(iabf, 0, imo2), ldb,
				Tdata(0.0), ptr_out, ldc);
			lda = nband;
			ldb = nw;
			Blas_Interface::gemm(
				'C', 'T',
				nmo1, nmo2, nw,
				Tdata(1.0), &Cs_ao_mo_k1(iabf, 0, imo1), lda, &psi_k2(imo2, 0), ldb,
				Tdata(1.0), ptr_out, ldc);
		}
	}
	else
	{
		std::vector<Tdata> Cs_ao_mo_k2_conj(nabf * nw * nmo2);
		for (std::size_t iabf = 0; iabf < nabf; ++iabf)
			for (std::size_t iw = 0; iw < nw; ++iw)
				for (std::size_t m2 = 0; m2 < nmo2; ++m2)
				{
					Cs_ao_mo_k2_conj[iabf * (nw * nmo2) + iw * nmo2 + m2]
						= Global_Func::get_conj(Cs_ao_mo_k2(iabf, iw, imo2 + m2));					
				}

		for (std::size_t iabf = 0; iabf < nabf; ++iabf)
		{
			Tdata *ptr_out = &Cs_mo(iabf, 0, 0);
			int lda = nw;
			int ldb = nmo2;
			int ldc = nmo2;
			Blas_Interface::gemm(
				'N', 'N',
				nmo1, nmo2, nw,
				Tdata(1.0), &psi_k1(imo1, 0), lda, &Cs_ao_mo_k2_conj[iabf * (nw * nmo2)], ldb,
				Tdata(0.0), ptr_out, ldc);
			lda = nband;
			ldb = nw;
			Blas_Interface::gemm(
				'T', 'C',
				nmo1, nmo2, nw,
				Tdata(1.0), &Cs_ao_mo_k1(iabf, 0, imo1), lda, &psi_k2(imo2, 0), ldb,
				Tdata(1.0), ptr_out, ldc);
		}
	}

	return Cs_mo;
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
std::map<Tk, std::map<Tk, Tensor<Tdata>>> 
LRI<TA, Tcell, Ndim, Tdata>::cal_cvc_mo_k_onthefly(
	const std::map<Tk, std::map<TA, Tensor<Tdata>>>& Cs_ao_mo, // C^mu (s,m)[k]
	const std::map<Tk, std::map<TA, Tensor<Tdata>>>& map_psi,  // c(m,s)[k]
	const std::vector<Tk>& k1_list,
	const std::vector<Tk>& k2_list,
	const std::vector<TA>& list_I,
	const std::vector<TA>& list_J,
	const Cell_Nearest<TA, Tcell, Ndim, double, 3>& cell_nearest,
	const std::vector<std::string>& psi_type,
	const std::size_t nocc,
	const std::size_t nvirt,
	const std::string& save_name,
	std::ostream& ofs,
	const std::vector<std::size_t>& order)
{
	using namespace Array_Operator;
#ifdef __MKL_RI
	const std::size_t mkl_threads = mkl_get_max_threads();
	mkl_set_num_threads(1);
#endif

	std::map<Tk, std::map<Tk, Tensor<Tdata>>> cvc_mo_k;
	std::map<Tk, std::map<std::pair<TA,TA>, Tensor<Tdata>>, Tk_Comparator> Vqs_fuzzy;

	const std::map<TA, std::map<TAC, Tensor<Tdata>>>& Vs = this->data_pool.at(save_name).Ds_ab;

	std::set<Tk, Tk_Comparator> q_set;
	Tk k_unit{1.0, 1.0, 1.0};
    for (const Tk& k1 : k1_list)
        for (const Tk& k2 : k2_list)
            q_set.insert( (k2 - k1) % k_unit );
	std::vector<Tk> q_list(q_set.begin(), q_set.end());
	print_k(ofs, q_list, "q_list");

	// add thread lock for the first Tk key of cvc_mo_k
	std::map<Tk, omp_lock_t> lock_cvc_result_add_map = LRI_Cal_Aux::init_lock_result(cvc_mo_k, k1_list);
	// add thread lock for the Tk key of Vq
	std::map<Tk, omp_lock_t> lock_vq_result_add_map = LRI_Cal_Aux::init_lock_result(Vqs_fuzzy, q_list);
	const std::vector<TAC> list_JR = Divide_Atoms::traversal_atom_period(list_J, this->period);
	
	#pragma omp parallel
	{
		// 1. FT V_mu_nu <I,<J,R>> to V_mu_nu <q,<I,J>>
		std::map<Tk, std::map<std::pair<TA,TA>, Tensor<Tdata>>, Tk_Comparator> Vqs_thread;
#pragma omp for schedule(dynamic) collapse(3)
		for (Tk q : q_list)
		{
			auto& Vq_thread = Vqs_thread[q];
			for (const TA mu: list_I)
			{
				const auto& V_mu = Vs.at(mu);
				for (const TAC& nu_R : list_JR)
				{
					const Tensor<Tdata>& V_mu_nu_R = Global_Func::find(V_mu, nu_R);
					if (V_mu_nu_R.empty()) continue;
					const TA nu = nu_R.first;
					const TC R_original = nu_R.second;
					double dist;
					const TC R = cell_nearest.cell_nearest_check(mu, nu, R_original, dist);
					// if (R_original != R)
					// {
					// 	#pragma omp critical
					// 	std::cout << "in cal_cvc_mo: cell_nearest_check gives different R from ("
					// 		<< R_original[0] << "," << R_original[1] << "," << R_original[2] << ") to ("
					// 		<< R[0] << "," << R[1] << "," << R[2] 
					// 		<< ") for I=" << mu << ", J=" << nu << ", dist=" << dist << std::endl;
					// }
					double arg = 2.0 * M_PI * (q[0] * R[0] + q[1] * R[1] + q[2] * R[2]);
					std::complex<double> fac (cos(arg), sin(arg));
					LRI_Cal_Aux::FT_Ds(V_mu_nu_R, Vq_thread[std::make_pair(mu, nu)], Global_Func::convert<Tdata>(fac));
				}
			}
			LRI_Cal_Aux::add_Ds_omp_try_map(Vqs_thread, Vqs_fuzzy, lock_vq_result_add_map, 1.0);
		}
		LRI_Cal_Aux::add_Ds_omp_wait_map(Vqs_thread, Vqs_fuzzy, lock_vq_result_add_map, 1.0);

		#pragma omp barrier
		#pragma omp master
		{
			LRI_Cal_Aux::destroy_lock_result(lock_vq_result_add_map, Vqs_fuzzy);
		}
		#pragma omp barrier

		// 2 calculate CVC_mo_k
		std::map<Tk, std::map<Tk, Tensor<Tdata>>> cvc_mo_k_thread;
#pragma omp for schedule(dynamic) collapse(4)
		for (const Tk k1: k1_list)
		{
			for (const Tk k2: k2_list)
			{
				Tk q = (k2-k1) % k_unit;
				const auto& Vq = Vqs_fuzzy.at(q);
				for (const TA mu : list_I)
				{
					// 2.1 calculate C^\mu (m1^*,m2)[k2,k1] on-the-fly, C_mu_ji for A and C_mu_bi for B
					const Tensor<Tdata> C_mu_ji = Cs_ao_mo_to_Cs_mo(Cs_ao_mo, map_psi, k2, k1, mu, psi_type[0], psi_type[1], nocc, nvirt, true);
					for (const TA nu : list_J)
					{
						const Tensor<Tdata>& Vq_mu_nu = Global_Func::find(Vq, std::make_pair(mu, nu));
						if (Vq_mu_nu.empty()) continue;

						// 2.2 calculate C^nu (m3,m4^*)[k2,k1] on-the-fly, C_nu_ba for A and C_nu_ja for B
						const Tensor<Tdata> C_nu_ba = Cs_ao_mo_to_Cs_mo(Cs_ao_mo, map_psi, k2, k1, nu, psi_type[2], psi_type[3], nocc, nvirt, false);
						// 2.3 calculate CVC_mo
						// CV_{ji,nu} = C^mu_{ji} V_{mu,nu} | CV_{bi,nu} = C^mu_{bi} V_{mu,nu}
						const Tensor<Tdata> CV_ji_nu = Tensor_Multiply::x1x2y1_ax1x2_ay1(C_mu_ji, Vq_mu_nu);
						const std::size_t nnu = Vq_mu_nu.shape[1];
						if (order == std::vector<std::size_t>{0,2,1,3}) // (jiba) -> (jbia)
						{	// (j,i nu) * (nu,b a) = (jiba) -> (j,bia)
							//    ̅            ̅                    ̅ ̅
							Tensor<Tdata> cvc({nocc, nvirt, nocc, nvirt});
							for (std::size_t b = 0; b < nvirt; ++b)
								for (std::size_t i = 0; i < nocc; ++i)
									{
										int lda = nnu * nocc;
										int ldb = nvirt * nvirt;
										int ldc = nvirt * nocc * nvirt;
										Blas_Interface::gemm(
											'N', 'N',
											nocc, nvirt, nnu,
											Tdata(1.0), &CV_ji_nu(0, i, 0), lda, &C_nu_ba(0, b, 0), ldb,
											Tdata(0.0), &cvc(0, b, i, 0), ldc);
									}
											
							LRI_Cal_Aux::add_Ds(std::move(cvc), cvc_mo_k_thread[k1][k2]);
						}
						else if (order == std::vector<std::size_t>{2,0,1,3}) // (bija) -> (jbia)
						{	// (b i,nu) * (nu,j a) = (bija) -> (jb,ia)
							//    ̅            ̅                  ̅   ̅ 
							Tensor<Tdata> cvc({nocc, nvirt, nocc, nvirt});
							for (std::size_t j = 0; j < nocc; ++j)
								for (std::size_t i = 0; i < nocc; ++i)
									{
										int lda = nnu * nocc;
										int ldb = nvirt * nocc;
										int ldc = nvirt * nocc;
										Blas_Interface::gemm(
											'N', 'N',
											nvirt, nvirt, nnu,/*CV_bi_nu*/      /*C_nu_ja*/
											Tdata(1.0), &CV_ji_nu(0, i, 0), lda, &C_nu_ba(0, j, 0), ldb,
											Tdata(0.0), &cvc(j, 0, i, 0), ldc);
									}
							LRI_Cal_Aux::add_Ds(std::move(cvc), cvc_mo_k_thread[k1][k2]);
						}
						else
						{
							throw std::runtime_error("Error in cal_cvc_mo_k_onthefly: unsupported order");
						}
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
	return cvc_mo_k;
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
std::map<Tk, std::map<Tk, Tensor<Tdata>>> 
LRI<TA, Tcell, Ndim, Tdata>::cal_cvc_mo_k_hartree_onthefly(
	const std::map<Tk, std::map<TA, Tensor<Tdata>>>& Cs_ao_mo, // C^mu (s,m)[k]
	const std::map<Tk, std::map<TA, Tensor<Tdata>>>& map_psi,  // c(m,t)[k]
	const std::vector<Tk>& k1_list,
	const std::vector<Tk>& k2_list,
	const std::vector<TA>& list_I,
	const std::vector<TA>& list_J,
	const std::vector<std::string>& psi_type,
	const std::size_t nocc,
	const std::size_t nvirt,
	const std::string& save_name,
	const bool is_A)
{
#ifdef __MKL_RI
	const std::size_t mkl_threads = mkl_get_max_threads();
	mkl_set_num_threads(1);
#endif

	std::map<Tk, std::map<Tk, Tensor<Tdata>>> cvc_mo_k;
	std::map<TA, std::map<TA, Tensor<Tdata>>> Vq; // has only one q=0, keep for further q ≠ 0 extension

	const std::map<TA, std::map<TAC, Tensor<Tdata>>>& Vs = this->data_pool.at(save_name).Ds_ab;

	// add thread lock for the first Tk key of cvc_mo_k
	std::map<Tk, omp_lock_t> lock_cvc_result_add_map = LRI_Cal_Aux::init_lock_result(cvc_mo_k, k1_list);
	// add thread lock for the TA key of Vq
	std::map<TA, omp_lock_t> lock_vq_result_add_map = LRI_Cal_Aux::init_lock_result(Vq, list_I);
	const std::vector<TAC> list_JR = Divide_Atoms::traversal_atom_period(list_J, this->period);
	#pragma omp parallel
	{
		Tk q{0.0, 0.0, 0.0};
		// 1. FT V_mu_nu <I,<J,R>> to V_mu_nu <q=0,<I,J>>
		std::map<TA, std::map<TA, Tensor<Tdata>>> Vq_thread;
#pragma omp for schedule(dynamic) collapse(2)
		for (const TA mu : list_I)
		{
			const auto& V_mu = Vs.at(mu);
			auto& Vq_mu_thread = Vq_thread[mu];
			for (const TAC& nu_R : list_JR)
			{
				const Tensor<Tdata>& V_mu_nu_R = Global_Func::find(V_mu, nu_R);
				if (V_mu_nu_R.empty()) continue;
				const TA nu = nu_R.first;
				const TC R = nu_R.second;
				double arg = 2.0 * M_PI * (q[0] * R[0] + q[1] * R[1] + q[2] * R[2]);
				std::complex<double> fac (cos(arg), sin(arg));
				LRI_Cal_Aux::FT_Ds(V_mu_nu_R, Vq_mu_thread[nu], Global_Func::convert<Tdata>(fac));
			}
			LRI_Cal_Aux::add_Ds_omp_try_map(Vq_thread, Vq, lock_vq_result_add_map, 1.0);
		}
		LRI_Cal_Aux::add_Ds_omp_wait_map(Vq_thread, Vq, lock_vq_result_add_map, 1.0);

		#pragma omp barrier
		#pragma omp master
		{
			LRI_Cal_Aux::destroy_lock_result(lock_vq_result_add_map, Vq);
		}
		#pragma omp barrier

		// 2 calculate CVC_mo_k
		std::map<Tk, std::map<Tk, Tensor<Tdata>>> cvc_mo_k_thread;
#pragma omp for schedule(dynamic) collapse(4)
		for (const TA mu : list_I)
		{
			const auto& Vq_mu = Vq.at(mu);
			for (const Tk k1: k1_list)
			{
				// 2.1 calculate C^\mu (i,a^*)[k1,k1] on-the-fly
				const Tensor<Tdata> C_mu_ia = Cs_ao_mo_to_Cs_mo(Cs_ao_mo, map_psi, k1, k1, mu, psi_type[0], psi_type[1], nocc, nvirt, false);
				for (const TA nu : list_J)
				{
					const Tensor<Tdata>& Vq_mu_nu = Global_Func::find(Vq_mu, nu);
					if (Vq_mu_nu.empty()) continue;
					for (const Tk k2: k2_list)
					{
						// 2.2 calculate C^nu (m3,m4)[k1,k2] on-the-fly, C_nu_j^*b for A and C_nu_jb^* for B
						Tensor<Tdata> C_nu_jb;
						if (is_A){
							C_nu_jb = Cs_ao_mo_to_Cs_mo(Cs_ao_mo, map_psi, k2, k2, nu, psi_type[2], psi_type[3], nocc, nvirt, true);
						}
						else{
							C_nu_jb = Cs_ao_mo_to_Cs_mo(Cs_ao_mo, map_psi, k2, k2, nu, psi_type[2], psi_type[3], nocc, nvirt, false);
						}
						// 2.3 calculate CVC_mo
						const Tensor<Tdata> CV_ia_nu = Tensor_Multiply::x1x2y1_ax1x2_ay1(C_mu_ia, Vq_mu_nu);
						LRI_Cal_Aux::add_Ds(Tensor_Multiply::x1x2y0y1_ax1x2_y0y1a(C_nu_jb, CV_ia_nu), cvc_mo_k_thread[k1][k2]);
					}
				}
				LRI_Cal_Aux::add_Ds_omp_try_map(cvc_mo_k_thread, cvc_mo_k, lock_cvc_result_add_map, 1.0);
			} // end for k1
		} // end for mu
		LRI_Cal_Aux::add_Ds_omp_wait_map(cvc_mo_k_thread, cvc_mo_k, lock_cvc_result_add_map, 1.0);
	} // end #pragma omp parallel

	LRI_Cal_Aux::destroy_lock_result(lock_cvc_result_add_map, cvc_mo_k);

	#ifdef __MKL_RI
	mkl_set_num_threads(mkl_threads);
	#endif

	malloc_trim(0);
	return cvc_mo_k;
}


// below are some functions reserved for reference
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

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
std::map<std::array<Tcell, Ndim>, std::map<std::array<Tcell, Ndim>, Tensor<Tdata>>> 
LRI<TA, Tcell, Ndim, Tdata>::cal_cvc_mo_R(
	const std::map<TA, std::map<std::pair<TC, TC>, Tensor<Tdata>>>& Cs_oo_mo,
	const std::map<TA, std::map<std::pair<TC, TC>, Tensor<Tdata>>>& Cs_vv_mo,
	const std::vector<TC>& R_list)
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

				const auto& Cs_mu_oo_ptr = Global_Func::find_map(Cs_oo_mo, mu);
				if (Cs_mu_oo_ptr.empty()) continue;
				const auto& Cs_nu_vv_ptr = Global_Func::find_map(Cs_vv_mo, nu);
				if (Cs_nu_vv_ptr.empty()) continue;
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
}	// end Lcal_cvc_mo_R

}	// end namespace RI

