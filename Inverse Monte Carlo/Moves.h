//
// Created by Joris on 09/07/2018.
//
#ifndef INVERSE_3D_NEW_MOVES_H
#define INVERSE_3D_NEW_MOVES_H
#endif //INVERSE_3D_NEW_MOVES_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include "Energy_changes.h"
using namespace Eigen;

std::uniform_int_distribution<int> unidir(0,2);
std::uniform_int_distribution<int> unidir_loop(1,5);

void kink_move(std::vector<Vector3i> &polymer, int site,int thread_num, int m){
    if ((polymer[(site+2)%pol_length] != polymer[site]) && (polymer[(site+2)%pol_length] != 2*polymer[(site+1)%pol_length]-polymer[site])){
        Vector3i prop_move1;
        prop_move1 = polymer[site] + polymer[(site+2)%pol_length] - polymer[(site+1)%pol_length];
        if (accept_move(delta_E_other(polymer, site, pol_length, prop_move1,thread_num))==1 && check_boundary_rest(prop_move1)==1 &&
                check_orient_rest(site, polymer, prop_move1)==1){
            //Throw away old contacts, at the same time update contact frequency map
            for (auto elem : locations[thread_num].find({polymer[(site+1)%pol_length][0],polymer[(site+1)%pol_length][1],polymer[(site+1)%pol_length][2]})->second ){
                if (elem != (site+1)%pol_length){
                    total_contacts[thread_num][std::min(elem,(site+1)%pol_length)][std::max(elem,(site+1)%pol_length)] += m - contacts[thread_num][{std::min(elem,(site+1)%pol_length),std::max(elem,(site+1)%pol_length)}];
                    contacts[thread_num].erase({std::min(elem,(site+1)%pol_length),std::max(elem,(site+1)%pol_length)});
                }
            }
            //put in new contacts
            if (locations[thread_num].find({prop_move1[0],prop_move1[1],prop_move1[2]})!=locations[thread_num].end()){
                for (auto elem : locations[thread_num].find({prop_move1[0],prop_move1[1],prop_move1[2]})->second ) {
                    contacts[thread_num][{std::min(elem,(site+1)%pol_length),std::max(elem,(site+1)%pol_length)}] = m;
                }
            }
            //update hash map locations
            std::vector<int> first_monomer = {polymer[((site+1)%pol_length)][0],polymer[((site+1)%pol_length)][1],polymer[((site+1)%pol_length)][2]};
            if (locations[thread_num][first_monomer].size() ==1){
                locations[thread_num].erase(first_monomer);
            }
            else {
                locations[thread_num][first_monomer].erase(std::find(locations[thread_num][first_monomer].begin(), locations[thread_num][first_monomer].end(),(site+1)%pol_length));
            }
            if (locations[thread_num].find({prop_move1[0],prop_move1[1],prop_move1[2]}) != locations[thread_num].end()){
                locations[thread_num][{prop_move1[0],prop_move1[1],prop_move1[2]}].push_back ((site+1)%pol_length);
            }
            else {
                locations[thread_num][{prop_move1[0],prop_move1[1],prop_move1[2]}] = {((site+1)%pol_length)};
            }

            //update polymer
            polymer[(site+1)%pol_length] = prop_move1;
        }
    }
}

void crankshaft_move(std::vector<Vector3i> &polymer, int site, int thread_num,int m){
    if ((polymer[(site+2)%pol_length] != polymer[site]) && (polymer[(site+2)%pol_length] != 2*polymer[(site+1)%pol_length]-polymer[site]) && (polymer[(site+3)%pol_length]-polymer[(site+2)%pol_length]==polymer[site]-polymer[(site+1)%pol_length])){
        int direction = unidir(gen);
        Vector3i prop_move1; Vector3i prop_move2;
        if (direction==1){ //180 degree flip
            prop_move1 = 2*polymer[site] - polymer[(site+1)%pol_length];
            prop_move2 = 2*polymer[(site+3)%pol_length] - polymer[(site+2)%pol_length];
        }
        else { //90 degree flip
            prop_move1 = polymer[site] + (direction-1)*(polymer[(site+1)%pol_length]-polymer[site]).cross(polymer[(site+3)%pol_length]-polymer[site]);
            prop_move2 = polymer[(site+3)%pol_length] + (direction-1)*(polymer[(site+2)%pol_length]-polymer[(site+3)%pol_length]).cross(polymer[(site+3)%pol_length]-polymer[site]);;
        }
        if (accept_move(delta_E_crankshaft(pol_length, polymer, site, prop_move1, prop_move2,thread_num))==1 && check_boundary_crankshaft(prop_move1, prop_move2)==1 &&
                check_orient_crankshaft(site, polymer, prop_move1, prop_move2)==1){
            //Throw away old contacts, at the same time update contact frequency map
            for (auto elem : locations[thread_num].find({polymer[(site+1)%pol_length][0],polymer[(site+1)%pol_length][1],polymer[(site+1)%pol_length][2]})->second ){
                if (elem != (site+1)%pol_length){
                    //std::cout << total_contacts[std::min(elem,(site+1)%pol_length)][std::max(elem,(site+1)%pol_length)] << '\t' << m << std::endl;
                    total_contacts[thread_num][std::min(elem,(site+1)%pol_length)][std::max(elem,(site+1)%pol_length)] += m - contacts[thread_num][{std::min(elem,(site+1)%pol_length),std::max(elem,(site+1)%pol_length)}];
                    contacts[thread_num].erase({std::min(elem,(site+1)%pol_length),std::max(elem,(site+1)%pol_length)});
                }
            }
            for (auto elem : locations[thread_num].find({polymer[(site+2)%pol_length][0],polymer[(site+2)%pol_length][1],polymer[(site+2)%pol_length][2]})->second ){
                if (elem != (site+2)%pol_length){
                    total_contacts[thread_num][std::min(elem,(site+2)%pol_length)][std::max(elem,(site+2)%pol_length)] += m - contacts[thread_num][{std::min(elem,(site+2)%pol_length),std::max(elem,(site+2)%pol_length)}];
                    contacts[thread_num].erase({std::min(elem,(site+2)%pol_length),std::max(elem,(site+2)%pol_length)});
                }
            }
            //put in new contacts
            if (locations[thread_num].find({prop_move1[0],prop_move1[1],prop_move1[2]})!=locations[thread_num].end()){
                for (auto elem : locations[thread_num].find({prop_move1[0],prop_move1[1],prop_move1[2]})->second ) {
                    contacts[thread_num][{std::min(elem,(site+1)%pol_length),std::max(elem,(site+1)%pol_length)}] = m;
                }
            }
            if (locations[thread_num].find({prop_move2[0],prop_move2[1],prop_move2[2]})!=locations[thread_num].end()){
                for (auto elem : locations[thread_num].find({prop_move2[0],prop_move2[1],prop_move2[2]})->second ) {
                    contacts[thread_num][{std::min(elem,(site+2)%pol_length),std::max(elem,(site+2)%pol_length)}] = m;
                }
            }
            // update hash map locations
            std::vector<int> first_monomer = {polymer[((site+1)%pol_length)][0],polymer[((site+1)%pol_length)][1],polymer[((site+1)%pol_length)][2]};
            std::vector<int> second_monomer = {polymer[((site+2)%pol_length)][0],polymer[((site+2)%pol_length)][1],polymer[((site+2)%pol_length)][2]};
            if (locations[thread_num][first_monomer].size() ==1){
                locations[thread_num].erase(first_monomer);
            }
            else {
                locations[thread_num][first_monomer].erase(std::find(locations[thread_num][first_monomer].begin(), locations[thread_num][first_monomer].end(),(site+1)%pol_length));
            }
            if (locations[thread_num][second_monomer].size() ==1){
                locations[thread_num].erase(second_monomer);
            }
            else {
                locations[thread_num][second_monomer].erase(std::find(locations[thread_num][second_monomer].begin(), locations[thread_num][second_monomer].end(),(site+2)%pol_length));
            }

            if (locations[thread_num].find({prop_move1[0],prop_move1[1],prop_move1[2]}) != locations[thread_num].end()){
                locations[thread_num][{prop_move1[0],prop_move1[1],prop_move1[2]}].push_back ((site+1)%pol_length);
            }
            else {
                locations[thread_num][{prop_move1[0],prop_move1[1],prop_move1[2]}] = {((site+1)%pol_length)};
            }
            if (locations[thread_num].find({prop_move2[0],prop_move2[1],prop_move2[2]}) != locations[thread_num].end()){
                locations[thread_num][{prop_move2[0],prop_move2[1],prop_move2[2]}].push_back (((site+2)%pol_length));
            }
            else {
                locations[thread_num][{prop_move2[0],prop_move2[1],prop_move2[2]}] = {((site+2)%pol_length)};
            }
            //update polymer
            polymer[(site+1)%pol_length] = prop_move1; polymer[(site+2)%pol_length] = prop_move2;
        }
    }
}

void loop_move(std::vector<Vector3i> &polymer, int site, int thread_num, int m){
    if (polymer[site] == polymer[(site+2)%pol_length]){
        int direction = unidir_loop(gen);
        Vector3i rotated_vector(3); Vector3i prop_move1;
        for (int i = 0; i<3;i++){ // rotate loop in one of 5 possible new directions
            rotated_vector[(i+direction/2)%3] = (-2*(direction%2)+1)*(polymer[(site+1)%pol_length]-polymer[site])[i];
        }
        prop_move1 = polymer[site] + rotated_vector;
        if (accept_move(delta_E_other(polymer, site, pol_length, prop_move1,thread_num))==1 && check_boundary_rest(prop_move1)==1 &&
                check_orient_rest(site, polymer, prop_move1)==1){
            //Throw away old contacts, at the same time update contact frequency map
            for (auto elem : locations[thread_num].find({polymer[(site+1)%pol_length][0],polymer[(site+1)%pol_length][1],polymer[(site+1)%pol_length][2]})->second ){
                if (elem != (site+1)%pol_length){
                    total_contacts[thread_num][std::min(elem,(site+1)%pol_length)][std::max(elem,(site+1)%pol_length)] += m - contacts[thread_num][{std::min(elem,(site+1)%pol_length),std::max(elem,(site+1)%pol_length)}];
                    contacts[thread_num].erase({std::min(elem,(site+1)%pol_length),std::max(elem,(site+1)%pol_length)});
                }
            }
            //put in new contacts
            if (locations[thread_num].find({prop_move1[0],prop_move1[1],prop_move1[2]})!=locations[thread_num].end()){
                for (auto elem : locations[thread_num].find({prop_move1[0],prop_move1[1],prop_move1[2]})->second ) {
                    contacts[thread_num][{std::min(elem,(site+1)%pol_length),std::max(elem,(site+1)%pol_length)}] = m;
                }
            }
            // update hash map locations
            std::vector<int> first_monomer = {polymer[((site+1)%pol_length)][0],polymer[((site+1)%pol_length)][1],polymer[((site+1)%pol_length)][2]};
            if (locations[thread_num][first_monomer].size() ==1){
                locations[thread_num].erase(first_monomer);
            }
            else {
                locations[thread_num][first_monomer].erase(std::find(locations[thread_num][first_monomer].begin(), locations[thread_num][first_monomer].end(),(site+1)%pol_length));
            }
            if (locations[thread_num].find({prop_move1[0],prop_move1[1],prop_move1[2]}) != locations[thread_num].end()){
                locations[thread_num][{prop_move1[0],prop_move1[1],prop_move1[2]}].push_back ((site+1)%pol_length);
            }
            else {
                locations[thread_num][{prop_move1[0],prop_move1[1],prop_move1[2]}] = {((site+1)%pol_length)};
            }
            //update polymer
            polymer[(site+1)%pol_length] = prop_move1;
        }
    }
}