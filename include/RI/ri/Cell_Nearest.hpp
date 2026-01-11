// ===================
//  Author: Peize Lin
//  date: 2022.12.30
// ===================

#pragma once

#include "Cell_Nearest.h"

#include "../global/Array_Operator.h"
#include "../global/Tensor_Algorithm.h"

#include <cmath>

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tpos, std::size_t Npos>
void Cell_Nearest<TA,Tcell,Ndim,Tpos,Npos>::init(
	const std::map<TA,Tatom_pos> &atoms_pos,
	const std::array<Tatom_pos,Ndim> &latvec_in,
	const std::array<Tcell,Ndim> &period_in)
{
	using namespace Array_Operator;
	this->period = period_in;

	for(std::size_t idim=0; idim<Ndim; ++idim)
		for(std::size_t ipos=0; ipos<Npos; ++ipos)
			this->latvec(idim,ipos) = latvec_in[idim][ipos];
	const Tensor<Tpos> least_square_tmp				// shape:{Ndim,Npos}
		= - Tensor_Algorithm::inverse_matrix_heev(latvec * latvec.transpose()) * latvec;

	for(const auto &atoms_pos_x : atoms_pos)
	{
		const TA &Ax = atoms_pos_x.first;
		for(const auto &atoms_pos_y : atoms_pos)
		{
			const TA &Ay = atoms_pos_y.first;
			const Tensor<Tpos> delta_pos = to_Tensor(atoms_pos_y.second - atoms_pos_x.second);
			this->cells_nearest_continuous[Ax][Ay] = to_array<Tpos,Ndim>(least_square_tmp * delta_pos);
		}
	}
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tpos, std::size_t Npos>
auto Cell_Nearest<TA,Tcell,Ndim,Tpos,Npos>::get_cell_nearest_discrete(
	const TA &Ax, const TA &Ay, const TC &cell) const
-> TC
{
	const std::array<Tpos,Ndim> &cell_nearest_continuous = this->cells_nearest_continuous.at(Ax).at(Ay);
	TC cell_nearest_discrete;
	for(std::size_t idim=0; idim<Ndim; ++idim)
		cell_nearest_discrete[idim]
			=  std::round( (cell_nearest_continuous[idim]-cell[idim]) / this->period[idim] )
				* this->period[idim]
				+ cell[idim];
	return cell_nearest_discrete;
}

template <typename TA, typename Tcell, std::size_t Ndim, typename Tpos, std::size_t Npos>
auto Cell_Nearest<TA, Tcell, Ndim, Tpos, Npos>::cell_nearest_check(
	const TA Ax, const TA Ay, const TC &cell, double &dist_min) const
-> TC
{
	static_assert(Ndim == 3, "cell_nearest_check currently assumes Ndim==3.");

    TC R_near = cell;
    const std::array<Tpos,Ndim> &Ryx = this->cells_nearest_continuous.at(Ax).at(Ay); //frac coordinate, Rx-Ry
    TC R_try;
    Tensor<Tpos> diff({Ndim}); // frac coordinate, pos_y - pos_x
	
	for (std::size_t i = 0; i < Ndim; ++i)
		diff(i) = R_near[i] - Ryx[i];
	int a_min(0), b_min(0), c_min(0);
	dist_min = (diff * this->latvec).norm(2);

    for (int a = -2; a < 3; ++a)
    {
        R_try[0] = a * this->period[0] + cell[0];
		diff(0) = R_try[0] - Ryx[0];
        for (int b = -2; b < 3; ++b)
        {
            R_try[1] = b * this->period[1] + cell[1];
			diff(1) = R_try[1] - Ryx[1];
            for (int c = -2; c < 3; ++c)
            {
                R_try[2] = c * this->period[2] + cell[2];
				diff(2) = R_try[2] - Ryx[2];
                double dist = (diff * this->latvec).norm(2);
                if (dist < dist_min - 1e-6)
                {
                    dist_min = dist;
                    R_near = R_try;
					a_min = a; b_min = b; c_min = c;
                }
				else if (std::abs(dist - dist_min) < 1e-6)
				{
					// in case of tie, choose the one with smaller abs(R-cell).z, then .y, then .x
					if (std::abs(c) < std::abs(c_min) || 
						(std::abs(c) == std::abs(c_min) && (std::abs(b) < std::abs(b_min) ||
						 (std::abs(b) == std::abs(b_min) && std::abs(a) < std::abs(a_min) ))))
					{
						dist_min = dist;
						R_near = R_try;
						a_min = a; b_min = b; c_min = c;
					}
				}
            }
        }
    }
    return R_near;
}
}
