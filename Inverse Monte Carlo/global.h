//
// Created by Joris on 13/07/2018.
//
#ifndef INVERSE_3D_NEW_GOBAL_H
#define INVERSE_3D_NEW_GOBAL_H


#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unordered_map>
#include <boost/functional/hash.hpp>

extern const int pol_length;
extern long int mc_moves;
extern const int burn_in_time;
extern const int cap;

//extern static std::mt19937_64 gen;
extern std::uniform_real_distribution<double> unif;
extern std::uniform_int_distribution<int> unimove;
extern std::uniform_int_distribution<int> unisite;

extern const int length_cylinder;
extern const int cap_length;
extern const int midway;

extern std::vector<std::vector<Eigen::Vector3i>> polymer;
extern std::vector<std::vector<std::vector<double>>> total_contacts;
extern const int number_of_threads;
extern std::vector<std::vector<int>> contacts_list;
extern std::vector<std::vector<int>> prop_contacts_list;

extern std::vector< std::vector<double>> Interaction_E;

extern bool boundary_cond;
extern bool orient;
extern const double diameter;
extern const double radius;
extern const int length_cell;
extern const int midway;

struct pair_hash {
    std::size_t operator()(const std::pair<int, int>& pair) const noexcept {
        std::size_t seed = 0;
        boost::hash_combine(seed, pair.first);
        boost::hash_combine(seed, pair.second);
        return seed;
    }
};
struct vec_hash {
    std::size_t operator()(const std::vector<int> vec) const noexcept {
        return boost::hash_range(vec.cbegin(), vec.cend());
    }
};
extern std::vector<std::unordered_map<std::pair<int, int>, int, pair_hash>> contacts;
extern std::vector<std::unordered_map<std::vector<int>, std::vector<int>, vec_hash>> locations;

#endif //INVERSE_3D_NEW_GOBAL_H

