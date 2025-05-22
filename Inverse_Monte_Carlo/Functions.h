//
// Created by Joris on 13/07/2018.
//
#ifndef INVERSE_3D_NEW_FUNCTIONS_H
#define INVERSE_3D_NEW_FUNCTIONS_H
#endif //INVERSE_3D_NEW_FUNCTIONS_H
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include "global.h"

static std::mt19937_64 gen(time(0));

bool accept_move(double delta_E){
    return ((delta_E <=0)||(unif(gen) < exp(-delta_E)));
}

// shape: chosen to match typical newborn C.crescentus cell

double x_centre_doubled = 6;
double y_centre_doubled = 6;
double diam_first = 2.85;
double diam_second= 4.5;
bool check_boundary_crankshaft(Eigen::Vector3i prop_move1, Eigen::Vector3i prop_move2){
    if (boundary_cond==1) {
        bool accept_1;
        bool accept_2;
        if (cap_length<=prop_move1[2] && prop_move1[2]< length_cylinder +cap_length) { //located inside cylinder?
            accept_1 = (pow(2 * prop_move1[0] - x_centre_doubled, 2) + pow(2 * prop_move1[1] - y_centre_doubled, 2) <=
                        pow(diameter, 2));
        }
        else if (prop_move1[2]==1 || prop_move1[2]==length_cylinder+cap_length) { //located in second ring?
            accept_1 = (pow(2*prop_move1[0]-x_centre_doubled, 2) + pow(2*prop_move1[1]-y_centre_doubled, 2) <= pow(diam_second, 2));
        }
        else if (prop_move1[2]==0 || prop_move1[2]==length_cylinder+cap_length+1) { //located in first ring?
            accept_1 = (pow(2*prop_move1[0]-x_centre_doubled, 2) + pow(2*prop_move1[1]-y_centre_doubled, 2) <= pow(diam_first, 2));
        } else {
            accept_1 = 0;
        }

        if (cap_length<=prop_move2[2] && prop_move2[2]< length_cylinder +cap_length) { //located inside cylinder?
            accept_2 = (pow(2 * prop_move2[0] - x_centre_doubled, 2) + pow(2 * prop_move2[1] - y_centre_doubled, 2) <=
                        pow(diameter, 2));
        }
        else if (prop_move2[2]==1 || prop_move2[2]==length_cylinder+cap_length) { //located in second ring?
            accept_2 = (pow(2*prop_move2[0]-x_centre_doubled, 2) + pow(2*prop_move2[1]-y_centre_doubled, 2) <= pow(diam_second, 2));
        }
        else if (prop_move2[2]==0 || prop_move2[2]==length_cylinder+cap_length+1) { //located in first ring?
            accept_2 = (pow(2*prop_move2[0]-x_centre_doubled, 2) + pow(2*prop_move2[1]-y_centre_doubled, 2)  <= pow(diam_first, 2));
        } else {
            accept_2 = 0;
        }
        return accept_1 * accept_2;
    }
    else{return 1;}
};

bool check_boundary_rest(Eigen::Vector3i prop_move1) {
    if (boundary_cond == 1) {
        if (cap_length<=prop_move1[2] && prop_move1[2]< length_cylinder +cap_length) { //located inside cylinder?
            return (pow(2 * prop_move1[0] - x_centre_doubled, 2) + pow(2 * prop_move1[1] - y_centre_doubled, 2) <=
                        pow(diameter, 2));
        }
        else if (prop_move1[2]==1 || prop_move1[2]==length_cylinder+cap_length) { //located in second ring?
            return (pow(2*prop_move1[0]-x_centre_doubled, 2) + pow(2*prop_move1[1]-y_centre_doubled, 2) <= pow(diam_second, 2));
        }
        else if (prop_move1[2]==0 || prop_move1[2]==length_cylinder+cap_length+1) { //located in first ring?
            return (pow(2*prop_move1[0]-x_centre_doubled, 2) + pow(2*prop_move1[1]-y_centre_doubled, 2) <= pow(diam_first, 2));
        } else { return 0; }
    }
    else {return 1;}
};

int origin = 0;

bool check_orient_crankshaft(int site, std::vector<Eigen::Vector3i> &polymer, Eigen::Vector3i prop_move1,
                             Eigen::Vector3i prop_move2){
    if (orient==1) {
        if ((site + 1)%pol_length == origin && prop_move1[2] >midway && polymer[(site + 1)%pol_length][2]<=midway)
        {
            return 0;
        } else if ((site + 2)%pol_length == origin && prop_move2[2] >midway && polymer[(site + 2)%pol_length][2]<=midway) {
            return 0;
        } else { return 1; }
    }
    else {return 1;}
}

bool check_orient_rest(int site, std::vector<Eigen::Vector3i> &polymer, Eigen::Vector3i prop_move1){
    if (orient==1) {
        if ((site + 1)%pol_length == origin && prop_move1[2] >midway && polymer[(site + 1)%pol_length][2]<=midway)
        {
            return 0;
        } else { return 1; }
    }
    else {return 1;}
}


