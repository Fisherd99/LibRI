#pragma once
#include "./Exx.h"

namespace RI
{
// Difference from Exx: 
// 1. in the `cal_force` function, two density matrices can be different
// 2. set_Cs and set_Vs can parallelize the input tensors according to listI and listJ
// 3. add set_Ws and free_Ws to setup screened Coulomb interaction
// 4. feat: cvc_mo in k space
template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
class  LR : public Exx<TA,Tcell,Ndim,Tdata>
{
public:
	using TC = std::array<Tcell,Ndim>;
	using TAC = std::pair<TA,TC>;
	using Tdata_real = Global_Func::To_Real_t<Tdata>;
	
	LR(const std::string method = "cvc") { this->method = method; }	// Exx default mehtod: "loop3"
	LR(Exx<TA, Tcell, Ndim, Tdata>&& exx) : Exx<TA, Tcell, Ndim, Tdata>::Exx(std::move(exx)) {}
	using Exx<TA,Tcell,Ndim,Tdata>::cal_force;
	// New function: cal_force with two different density matrices
	void cal_force(const std::map<TA, std::map<std::pair<TA, std::array<Tcell, Ndim>>, Tensor<Tdata>>>& Ds_left,
		const std::array<std::string, 5>& save_names_suffix = { "","","","","" })	// "Cs","Vs","Ds","dCs","dVs"
	{
		// The only difference from Exx::cal_force: save Ds_left if not empty
		if (!Ds_left.empty())
			this->post_2D.saves["Ds_" + save_names_suffix[2]] = this->post_2D.set_tensors_map2(Ds_left);
		this->cal_force(save_names_suffix);
	}

	void set_Cs(
		std::map<TA, std::map<TAC, Tensor<Tdata>>> &Cs,
		const Tdata_real &threshold,
		const std::set<TA> &listI,
		const std::set<TA> &listJ,
		const std::string &save_name_suffix="")
	{
		Cs = Communicate_Tensors_Map_Judge::comm_map2_first(this->lri.mpi_comm, std::move(Cs), listI, listJ);
		this->lri.set_tensors_map2(
			Cs,
			{Label::ab::a, Label::ab::b},
			{{"threshold_filter", threshold}, {"flag_comm", false}},
			"Cs_"+save_name_suffix );
		this->flag_finish.Cs = true;
	}

	void set_Vs(
		std::map<TA, std::map<TAC, Tensor<Tdata>>> &Vs,
		const Tdata_real &threshold,
		const std::set<TA> &listI,
		const std::set<TA> &listJ,
		const std::string &save_name_suffix="")
	{
		Vs = Communicate_Tensors_Map_Judge::comm_map2_first(this->lri.mpi_comm, std::move(Vs), listI, listJ);
		this->lri.set_tensors_map2(
			Vs,
			{Label::ab::a0b0},
			{{"threshold_filter", threshold}, {"flag_comm", false}},
			"Vs_"+save_name_suffix );
		this->flag_finish.Vs = true;
	};
	
	// setup screened Coulomb interaction and parallelize according to listI and listJ
	void set_Ws(
		std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ws,
		const Tdata_real &threshold,
		const std::set<TA> &listI,
		const std::set<TA> &listJ,
		const std::string &save_name_suffix="")
	{
		Ws = Communicate_Tensors_Map_Judge::comm_map2_first(this->lri.mpi_comm, std::move(Ws), listI, listJ);
		this->lri.set_tensors_map2(
			Ws,
			{Label::ab::a0b0},
			{{"threshold_filter", threshold}, {"flag_comm", false}},
			"Ws_"+save_name_suffix );
		this->flag_Ws = true;
	};

	void free_Ws(const std::string &save_name_suffix="")
	{
		this->lri.free_tensors_map2("Ws_"+save_name_suffix);
		this->flag_Ws = false;
	};

	bool flag_Ws = false;
};

}