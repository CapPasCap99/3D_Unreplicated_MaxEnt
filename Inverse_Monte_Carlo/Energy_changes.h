//
// Created by Joris on 11/07/2018.
//
#ifndef INVERSE_3D_NEW_ENERGY_CHANGES_H
#define INVERSE_3D_NEW_ENERGY_CHANGES_H
#endif //INVERSE_3D_NEW_ENERGY_CHANGES_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "Functions.h"
#include "Initialize.h"
#include "global.h"
using namespace Eigen;

double delta_E_crankshaft(int pol_length, std::vector<Vector3i> &polymer, int site, Vector3i prop_move1, Vector3i prop_move2,int thread_num){
    double energy_change = 0;
    if (locations[thread_num].find({prop_move1[0],prop_move1[1],prop_move1[2]})!=locations[thread_num].end()){
        for (auto elem : locations[thread_num].find({prop_move1[0],prop_move1[1],prop_move1[2]})->second ) {
            energy_change += Interaction_E[elem][site+1];
        }
    }
    if (locations[thread_num].find({prop_move2[0],prop_move2[1],prop_move2[2]})!=locations[thread_num].end()){
        for (auto elem : locations[thread_num].find({prop_move2[0],prop_move2[1],prop_move2[2]})->second ) {
            energy_change += Interaction_E[elem][site+2];
        }
    }
    for (auto elem : locations[thread_num].find({polymer[(site+1)%pol_length][0],polymer[(site+1)%pol_length][1],polymer[(site+1)%pol_length][2]})->second ) {
        if (elem != (site+1)%pol_length){
            energy_change -= Interaction_E[(site+1)%pol_length][elem];
        }
    }
    for (auto elem : locations[thread_num].find({polymer[(site+2)%pol_length][0],polymer[(site+2)%pol_length][1],polymer[(site+2)%pol_length][2]})->second ) {
        if (elem != (site+2)%pol_length){
            energy_change -= Interaction_E[(site+2)%pol_length][elem];
        }
    }
    return energy_change;
}

double delta_E_other(std::vector<Vector3i> &polymer, int site, int pol_length, Vector3i prop_move1,int thread_num){
    double energy_change = 0;
    if (locations[thread_num].find({prop_move1[0],prop_move1[1],prop_move1[2]})!=locations[thread_num].end()){
        for (auto elem : locations[thread_num].find({prop_move1[0],prop_move1[1],prop_move1[2]})->second ) {
            energy_change += Interaction_E[elem][site+1];
        }
    }
    for (auto elem : locations[thread_num].find({polymer[(site+1)%pol_length][0],polymer[(site+1)%pol_length][1],polymer[(site+1)%pol_length][2]})->second ) {
        if (elem != (site+1)%pol_length){
            energy_change -= Interaction_E[(site+1)%pol_length][elem];
        }
    }
    return energy_change;
}

