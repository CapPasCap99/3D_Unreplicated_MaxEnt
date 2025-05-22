#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <unordered_map>
#include <thread>
#include "Initialize.h"
#include "Moves.h"
using namespace Eigen;

//const int number_of_threads = 36;
const int number_of_threads = 30;

const double diameter = 6.4;
const int length_cylinder = 21;
const int midway = int(ceil(length_cylinder/2));
const int cap_length = 2;

const int pol_length = 1620;

const long int mc_moves_start = 100000000;
long int mc_moves;
const int burn_in_time = 20000000;
//const int update_steps = 30;
const int update_steps=20;
const double learning_rate = 0.5;

std::vector<std::vector<Vector3i>> polymer(number_of_threads);

bool boundary_cond = 1; //enforces boundary conditions if 1
bool orient = 1; //orients the cell such that the origin is always in the left half

std::vector< std::vector<double>> Interaction_E(pol_length, std::vector<double>(pol_length,0));

std::uniform_real_distribution<double> unif(0.0,1.0);
std::uniform_int_distribution<int> unimove(0,2);
std::uniform_int_distribution<int> unisite(0,pol_length-1);
std::vector<std::vector<std::vector<double>>> total_contacts(number_of_threads, std::vector< std::vector<double>>(pol_length, std::vector<double>(pol_length, 0)));
std::vector<std::vector<double>> final_contacts(pol_length, std::vector<double>(pol_length, 0));

void move(std::vector<Vector3i> &polymer, int thread_num, int m) {
    int action = unimove(gen);
    int site = unisite(gen);
    if (action == 0) kink_move(polymer, site, thread_num, m);
    else if (action == 1) crankshaft_move(polymer, site, thread_num, m);
    else loop_move(polymer, site, thread_num, m);
}

void run_forward(int thread_num, int steps) {
    for (int m = 1; m < steps; m++) {
        move(polymer[thread_num], thread_num, m);
    }
}

int main() {
    std::cout << "Recovery simulation started!" << std::endl;

    // Step 1: Initialize polymers
    for (int i = 0; i < number_of_threads; i++) {
        initialize(polymer[i], pol_length, i);
    }

    // Step 2: Load saved intermediate energies
    std::ifstream energy_file("/home/capucine/Documents/test/Data/Intermediate_Energies/energies_intermediate_19.txt");
    if (!energy_file.is_open()) {
        std::cerr << "Failed to open energies_intermediate_19.txt" << std::endl;
        return 1;
    }

    for (int i = 0; i < pol_length; i++) {
        for (int j = 0; j < pol_length; j++) {
            energy_file >> Interaction_E[i][j];
        }
    }
    energy_file.close();
    std::cout << "Loaded saved energies." << std::endl;

    // Step 3: Forward MC simulation
    std::vector<std::thread> threads(number_of_threads);
    for (int i = 0; i < number_of_threads; i++) {
        threads[i] = std::thread(run_forward, i, burn_in_time);
    }
    for (auto &t : threads) t.join();

    std::cout << "Finished forward simulation." << std::endl;

    // Step 4: Save final configurations
    int n_capture = 0;
    while (n_capture < number_of_threads) {
        for (int thread_num = 0; thread_num < number_of_threads; thread_num++) {
            std::ofstream final_config;
            final_config.open("/home/capucine/Documents/test/Data/Final_Configurations/recovered_configuration_" +
                std::to_string(n_capture) + ".txt");
            for (int i = 0; i < pol_length; i++) {
                for (int j = 0; j < 3; j++) {
                    final_config << polymer[thread_num][i][j] << std::endl;
                }
            }
            final_config.close();
            n_capture++;

        }
    }

    for (int thread_num = 0; thread_num < number_of_threads; thread_num++) {
        if (polymer[thread_num][0][2] < 20) {
                std::ofstream final_config;
                final_config.open("/home/capucine/Documents/test/Data/Final_Configurations/recovered_configuration_" +
                                  std::to_string(n_capture) + ".txt");
                for (int i = 0; i < pol_length; i++) {
                        for (int j = 0; j < 3; j++) {
                                final_config << polymer[thread_num][i][j] << std::endl;
                        }
                 }
                 final_config.close();
        }
    }
    std::cout << "Recovery complete. Configurations saved!" << std::endl;
    return 0;

