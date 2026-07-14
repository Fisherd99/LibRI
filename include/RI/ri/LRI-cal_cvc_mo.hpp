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
	const std::vector<std::string>& psi_type,
	const std::size_t nocc,
	const std::size_t nvirt,
	const std::string& save_name,
	const bool is_A,
	const std::vector<Tk>& q_list,
	const std::map<Tk, std::vector<std::pair<Tk, Tk>>, Tk_Comparator>& q2kpair)
{
	using namespace Array_Operator;
#ifdef __MKL_RI
	const std::size_t mkl_threads = mkl_get_max_threads();
	mkl_set_num_threads(1);
#endif

	std::map<Tk, std::map<Tk, Tensor<Tdata>>> cvc_mo_k;
	const std::map<TA, std::map<TAC, Tensor<Tdata>>>& Vs = this->data_pool.at(save_name).Ds_ab;
	// add thread lock for the first Tk key of cvc_mo_k
	std::map<Tk, omp_lock_t> lock_cvc_result_add_map = LRI_Cal_Aux::init_lock_result(cvc_mo_k, k1_list);
#pragma omp parallel
	{
		// 2. calculate CVC_mo_k
		std::map<Tk, std::map<Tk, Tensor<Tdata>>> cvc_mo_k_thread;
#pragma omp for schedule(dynamic, 64) collapse(3)
		for (const Tk q: q_list)
		{
			for (const TA mu: list_I)
			{
				for (const TA nu: list_J)
				{	// 2.1 calculate V(q)_{mu,nu} on-the-fly
					Tensor<Tdata> Vq_mu_nu;
					const auto& V_mu = Vs.at(mu);
					const std::vector<TAC> list_nuR = Divide_Atoms::traversal_atom_period(std::vector<TA>{nu}, this->period);
					for (const TAC& nu_R : list_nuR)
					{
						const Tensor<Tdata>& V_mu_nu_R = Global_Func::find(V_mu, nu_R);
						if (V_mu_nu_R.empty()) continue;
						const TC& R = nu_R.second;
						double arg = 2.0 * M_PI * (q[0] * R[0] + q[1] * R[1] + q[2] * R[2]);
						std::complex<double> fac (cos(arg), sin(arg));
						LRI_Cal_Aux::add_Ds(V_mu_nu_R, Vq_mu_nu, Global_Func::convert<Tdata>(fac));
					}
					if (Vq_mu_nu.empty()) continue;

					for (const auto& kpair: q2kpair.at(q))
					{
						const Tk k1 = kpair.first;
						const Tk k2 = kpair.second;

						// 2.2 calculate C^\mu (m1^*,m2)[k2,k1] on-the-fly, C_mu_ji for A and C_mu_bi for B
						const Tensor<Tdata> C_mu_ji = Cs_ao_mo_to_Cs_mo(Cs_ao_mo, map_psi, k2, k1, mu, psi_type[0], psi_type[1], nocc, nvirt, true);
						// 2.3 calculate C^nu (m3,m4^*)[k2,k1] on-the-fly, C_nu_ba for A and C_nu_ja for B
						const Tensor<Tdata> C_nu_ba = Cs_ao_mo_to_Cs_mo(Cs_ao_mo, map_psi, k2, k1, nu, psi_type[2], psi_type[3], nocc, nvirt, false);
						// 2.4 calculate CVC_mo
						// CV_{ji,nu} = C^mu_{ji} V_{mu,nu} | CV_{bi,nu} = C^mu_{bi} V_{mu,nu}
						const Tensor<Tdata> CV_ji_nu = Tensor_Multiply::x1x2y1_ax1x2_ay1(C_mu_ji, Vq_mu_nu);
						const std::size_t nnu = Vq_mu_nu.shape[1];
						if (is_A)
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
						else
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
					} // end for kpair
					LRI_Cal_Aux::add_Ds_omp_try_map(cvc_mo_k_thread, cvc_mo_k, lock_cvc_result_add_map, 1.0);
				} // end for nu
			} // end for mu
		} // end for q
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
		for (const TA mu : list_I)
		{
			const auto& V_mu = Vs.at(mu);
			auto& Vq_mu_thread = Vq_thread[mu];
	#pragma omp for schedule(dynamic)
			for (const TAC& nu_R : list_JR)
			{
				const Tensor<Tdata>& V_mu_nu_R = Global_Func::find(V_mu, nu_R);
				if (V_mu_nu_R.empty()) continue;
				const TA nu = nu_R.first;
				const TC& R = nu_R.second;
				double arg = 2.0 * M_PI * (q[0] * R[0] + q[1] * R[1] + q[2] * R[2]);
				std::complex<double> fac (cos(arg), sin(arg));
				LRI_Cal_Aux::add_Ds(V_mu_nu_R, Vq_mu_thread[nu], Global_Func::convert<Tdata>(fac));
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
#pragma omp for schedule(dynamic, 64) collapse(2)
		for (const TA mu : list_I)
		{
			for (const Tk k1: k1_list)
			{
				// 2.1 calculate C^\mu (i,a^*)[k1,k1] on-the-fly
				const Tensor<Tdata> C_mu_ia = Cs_ao_mo_to_Cs_mo(Cs_ao_mo, map_psi, k1, k1, mu, psi_type[0], psi_type[1], nocc, nvirt, false);
				for (const TA nu : list_J)
				{
					const Tensor<Tdata>& Vq_mu_nu = Global_Func::find(Vq, mu, nu);
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

}	// end namespace RI

