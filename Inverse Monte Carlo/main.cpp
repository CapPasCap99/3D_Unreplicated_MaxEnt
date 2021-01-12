//
// Created by Joris on 09/07/2018.
//

// This code performs an iterative Monte Carlo procedure to obtain a Maximum Entropy
// model for the chromosome organization of C. crescentus


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

const int number_of_threads = 36;

const double diameter = 6.4;
const int length_cylinder = 21;
const int midway = int(ceil(length_cylinder/2));
const int cap_length = 2;

const int pol_length = 1620;

const long int mc_moves_start = 100000000;
long int mc_moves;
const int burn_in_time = 20000000;
const int update_steps = 30;
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

void move(std::vector<Vector3i> &polymer,int thread_num, int m){ //performs a single Monte Carlo step
    int action;
    int site;

    action = unimove(gen);
    site = unisite(gen);
    if (action==0){
        kink_move(polymer,site, thread_num,m);
    }
    else if (action==1){
        crankshaft_move(polymer,site, thread_num,m);
    }
    else if (action==2){
       loop_move(polymer,site, thread_num,m);
    }
}

void run_burnin(int thread_num, int mc_moves) { //burns in the polymer configurations
    for (int m = 1; m < mc_moves; m++) {
        move(polymer[thread_num], thread_num, m);
    }
}

void run(int thread_num, int mc_moves) {
    for (int m = 1; m < mc_moves; m++) {    //performs a forward polymer simulation
        move(polymer[thread_num], thread_num, m);
    }
}

//after each forward run, the polymer interaction energies are updated according to the pairwise difference
// between model contact frequencies and experimental contact frequencies
void update_energies(std::vector< std::vector<double>> &total_contacts, std::vector< std::vector<double>> &reference_contacts, std::vector< std::vector<double>> &Interaction_E) {

    float checksum = 0;
    for (int i = 0; i < pol_length/4; i++) {
        for (int j = i+1; j < pol_length/4; j++) {
            if ((i != (j+1)%(pol_length/4)) && (j != (i+1)%(pol_length/4))) {
                Interaction_E[4 * i][4 * j] += learning_rate * sqrt(1/(std::max(reference_contacts[4*i][4*j],0.0001)))*(float(total_contacts[4 * i][4 * j])- reference_contacts[4 * i][4 * j]);
                checksum += float(total_contacts[4 * i][4 * j])*(pol_length/4);
                Interaction_E[4 * j][4 * i] = Interaction_E[4 * i][4 * j];
            }
        }
    }

    //calculates the shift of all energies, imposed to ensure a MaxEnt solution is found for the contact frequency scale
    float shift = 0;
    for (int i = 0; i < pol_length/4; i++) {
        for (int j = i + 2; j < pol_length / 4; j++) {
           shift += 2*(Interaction_E[4 * i][4 * j] * reference_contacts[4 * i][4 * j]) / (pol_length / 4);
        }
    }

    std::cout << "Shift: " << shift << std::endl;
    for (int i = 0; i < pol_length/4; i++) {
        for (int j = i + 2; j < pol_length / 4; j++) {
            //if ((i != (j + 1) % (pol_length / 2)) && (j != (i + 1) % (pol_length / 2)) && not(i >=0 && i<=30 && j>=150 && j<=220) && not(i >=150 && i <=230 && j>=370 && j<=410) ) {
            Interaction_E[4 * i][4 * j] -= shift;
            Interaction_E[4 * j][4 * i] = Interaction_E[4 * i][4 * j];
            //}
        }
    }
}

//Normalizes model contact frequencies to allow for comparison with experimental data
void normalize() {
    double sum = 0;
    for (int i = 0; i < pol_length; i++) {
        for (int j = 0; j < pol_length; j++) {
            if (i%4 == 0 && j%4 == 0 && i!=(j+4)%pol_length && j!=(i+4)%pol_length) {
                sum += final_contacts[i][j];
            }
        }
    }
    for (int i = 0; i < pol_length; i++) {
        for (int j = 0; j < pol_length; j++) {
            final_contacts[i][j] *= double(pol_length)/(8*sum);
        }
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "started! " << std::endl;

    for (int l = 0; l < number_of_threads; l++) {
        initialize(polymer[l], pol_length, l);
    }
    std::cout << "Initialized " << std::endl;

    // Read in starting interaction energies
    std::ifstream couplings;
    couplings.open("/media/joris/raid_data/Chromosome_Maxent/Dataprocess_check_review/Processed_new_r0/Inverse/Energies_Intermediate/energies_intermediate_49.txt");
    //couplings.open ("/media/joris/raid_data/Chromosome_Maxent/Saved_energies/energies_smallcell.txt");
    for (int i = 0; i < pol_length; i++) { //read in energies
        for (int j = 0; j < pol_length; j++) {
            couplings >> Interaction_E[i][j];
        }
    }
    couplings.close();

    // Read in reference contacts
    std::vector<std::vector<double>> reference_contacts(pol_length, std::vector<double>(pol_length));
    std::ifstream input_contacts;
    input_contacts.open("/media/joris/raid_data/Chromosome_Maxent/experimental_data_noisecancel_r1.txt");
    for (int i = 0; i < pol_length / 4; i++) { //read in contacts
        for (int j = 0; j < pol_length / 4; j++) {
            input_contacts >> reference_contacts[4 * i][4 * j];
            if (reference_contacts[4*i][4*j] == 0 && i != j && i != (j + 1) % (pol_length / 2) &&
                j != (i + 1) % (pol_length / 2)) {
                Interaction_E[4 * i][4 * j] = 10;
            }
        }
    }
    input_contacts.close();

    // burn in configurations
    std::vector<std::thread> threads(number_of_threads);
    for (auto l = 0; l < number_of_threads; l++) {
        threads[l] = std::thread(run_burnin, l, burn_in_time);
    }
    for (auto &&l : threads) {
        l.join();
    }

    std::cout << "Done with burn in " << std::endl;

    for (int n =0; n<update_steps;n++) { //do iterative update scheme
        mc_moves = mc_moves_start*sqrt(n+10)/sqrt(10); //number of MC moves grows with each iteration (implicitly converted to long int)

        for (int l = 0; l < number_of_threads; l++) {
            for (int i = 0; i < pol_length; i++) { //reset contact frequencies before starting new forward round
                for (int j = 0; j < pol_length; j++) {
                    total_contacts[l][i][j] = 0;
                    if (l==0){
                        final_contacts[i][j] =0;
                    }
                }

            }
            for (auto elem : contacts[l]) { //reset contacts
                contacts[l][elem.first] = 0;
            }
        }

        //run forward simulation
        for (auto l = 0; l < number_of_threads; l++) {
            threads[l] = std::thread(run, l, mc_moves);
        }
        for (auto &&l : threads) {
            l.join();
        }
        // read in remaining contacts at the end of forward simulation
        for (auto l = 0; l < number_of_threads; l++) {
            for (auto elem : contacts[l]) { //add the contacts remaining at the end of the simulation
                total_contacts[l][elem.first.first][elem.first.second] += mc_moves - elem.second;
                contacts[l][elem.first] = 0;
            }
        }

        //add up contacts from threads
        for (int i = 0; i < number_of_threads; i++) {
            for (int j = 0; j < pol_length; j++) {
                for (int k = 0; k < pol_length; k++) {
                    final_contacts[j][k] += total_contacts[i][j][k];
                }
            }
        }

        //normalize contact frequencies
        normalize();

        //update energies
        update_energies(final_contacts,reference_contacts,Interaction_E);

        //output intermediate results
        std::ofstream intermediate_energies;
        intermediate_energies.open ("/media/joris/raid_data/Chromosome_Maxent/Dataprocess_check_review/Processed_new_r0/Inverse/Energies_Intermediate/energies_intermediate_" + std::to_string(n) + ".txt");
        for(int i = 0; i < pol_length; i++){ //output contact frequencies
            for(int j = 0; j < pol_length; j++){
                intermediate_energies << Interaction_E[i][j] << std::endl;
            }
        }
        intermediate_energies.close();

        std::ofstream intermediate_contacts;
        intermediate_contacts.open ("/media/joris/raid_data/Chromosome_Maxent/Dataprocess_check_review/Processed_new_r0/Inverse/Contacts_Intermediate/contacts_intermediate_" + std::to_string(n) + ".txt");
        for(int i = 0; i < pol_length; i++){ //output contact frequencies
            for(int j = 0; j < pol_length; j++){
                double contact = double(final_contacts[i][j]);
                intermediate_contacts << contact << std::endl;
            }
        }
        intermediate_contacts.close();
    }

    std::cout << "Finished! "  << std::endl;

    //output final configurations for each thread
    int n_capture = 0;
    while (n_capture <number_of_threads) {
        for (int thread_num = 0; thread_num < number_of_threads; thread_num++) {
            if (polymer[thread_num][0][2] < 20) {
                std::ofstream final_configuration;
                final_configuration.open(
                        "/media/joris/raid_data/Chromosome_Maxent/Dataprocess_check_review/Processed_new_r0/Inverse/configuration_init_" +
                        std::to_string(n_capture) + ".txt");
                for (int i = 0; i < pol_length; i++) { //output contact frequencies
                    for (int j = 0; j < 3; j++) {
                        final_configuration << polymer[thread_num][i][j] << std::endl;
                    }
                }
                final_configuration.close();
                n_capture++;
            }
        }
    }



    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " <<  elapsed.count() << " seconds\n";
    return 0;
}