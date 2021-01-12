//
// Created by Joris on 09/07/2018.
//
#ifndef Inverse_3D_new_Initialize_h
#define Inverse_3D_new_Initialize_h

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <string>
#include "global.h"
using namespace Eigen;

std::vector<std::unordered_map<std::pair<int, int>, int, pair_hash>> contacts(number_of_threads);
std::vector<std::unordered_map<std::vector<int>, std::vector<int>, vec_hash>> locations(number_of_threads);

void initialize(std::vector<Vector3i> &polymer, int pol_length, int thread_num){
        int coordinate;
        std::ifstream configuration;
        configuration.open ("/media/joris/raid_data/Chromosome_Maxent/Dataprocess_check_review/Raw/Inverse/configuration_init_" +std::to_string(thread_num)+".txt");
        for(int i = 0; i < pol_length; i++){
            Vector3i monomer;
            for(int j = 0; j < 3; j++){
                configuration >> coordinate;
                monomer(j) = coordinate;
            }
            polymer.push_back(monomer);
            if (locations[thread_num].find({monomer(0),monomer(1),monomer(2)})!=locations[thread_num].end()){
                for (auto elem : locations[thread_num].find({monomer(0),monomer(1),monomer(2)})->second){
                    contacts[thread_num][{elem,i}]=0;
                }
                locations[thread_num].find({monomer(0),monomer(1),monomer(2)})->second.push_back(i);
            }
            else { locations[thread_num][{monomer(0),monomer(1),monomer(2)}] = {i}; }
        }
    configuration.close();

//    int x_i, y_i, z_i;
//    Vector3i monomer;
//    x_i=2; y_i=2; z_i=2;
//    for (int i=0; i<pol_length; i++){
//            if (i%4==0){x_i++;}
//            else if (i%4==1){y_i++;}
//            else if (i%4==2){x_i--;}
//            else if (i%4==3){y_i--;}
//        monomer << x_i,y_i,z_i; polymer.push_back(monomer);
//        if (locations[thread_num].find({x_i,y_i,z_i})!=locations[thread_num].end()){
//            for (auto elem : locations[thread_num].find({x_i,y_i,z_i})->second){
//                contacts[thread_num][{elem,i}]=0;
//            }
//            locations[thread_num].find({x_i,y_i,z_i})->second.push_back(i);
//            }
//        else { locations[thread_num][{x_i,y_i,z_i}] = {i}; }
//    }
}
#endif //Inverse_3D_new_Initialize.h