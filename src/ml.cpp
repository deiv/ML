//
// Created by deiv on 8/04/18.
//

#include <iostream>
#include <sstream>
#include <cstddef>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <numeric>
#include <future>
#include <cmath>

#include "ml.h"

/*
 * Nombres de espacios lo mas limpios posibles ...
 */
using std::string;
using std::cout;
using std::endl;
using std::set;
using std::map;
using std::vector;
using std::pair;
using std::future;
using std::ref;

namespace ml {

result_tree_t* DecisionTree::create_decision_tree(col_idx_t attr_col_idx)
{
    set<csv_field_t> col_values;

    /*
     * Obtenemos los valores unicos para la columna indice
     */
    for (auto& s : *data) {
        col_values.insert(s->at(attr_col_idx));
    }

    /*
     * Calculamos la ganancia de información para la con más peso
     */
    set<col_idx_t> empty_set;
    result_tree_t* root_tree = calculate_ig(data, col_values, attr_col_idx, empty_set);

    return root_tree;
}


/*
 * IG(Y% X) = H(Y) − H(Y% X)
 */
result_tree_t* DecisionTree::calculate_ig(datacontainer_t &data, set<csv_field_t> values, col_idx_t attr_col_idx, set<col_idx_t>& ignored_cols)
{
    /*
     * Calculamos la entropia de la columna a predecir
     */
    double attr_entropy = calculate_dataset_entropy(data, values, attr_col_idx);

    /*
     * attr_entropy == 0 -> nodo hoja, finalizamos
     */
    if (attr_entropy == 0.0d) {
        std::vector<result_tree_t*> empty_vec;
        std::vector<result_tree_t*> vec;
        auto node_tree = new result_tree_t();
        auto node_tree_sheet = new result_tree_t();

        node_tree_sheet->children =  empty_vec;
        node_tree_sheet->attr_value = data->at(0)->at(attr_col_idx);
        node_tree_sheet->root = node_tree;

        vec.push_back(node_tree_sheet);
        node_tree->children =  vec;
        node_tree->attr_name = col_names->at(attr_col_idx);

        return node_tree;
    }

    size_t col_count = data->at(0)->size();
    map<col_idx_t, vector<dataset_entropy>> cols_entropies;
    vector<future<pair<col_idx_t, vector<dataset_entropy>>>> fut_vec;

    /*
     * Lanzamos el calculo de las entropias de cada columna en paralelo
     */
    for (col_idx_t col_idx = 0; col_idx < col_count; col_idx++) {
        if (col_idx == attr_col_idx) {
            continue;
        }

        if (ignored_cols.count(col_idx)) {
            continue;
        }

       /* fut_vec.push_back(
            std::async(
                std::launch::async,
                calculate_dataset_entropy_2_col,
                ref(data),
                values,
                attr_col_idx,
                col_idx));*/

        cols_entropies.insert(calculate_dataset_entropy_2_col(data, values, attr_col_idx, col_idx));
    }

    /* XXX: activar hilos */
    /*
     * Bloqueamos hasta obtener los resultados del calculo de entropias
     */
    /*for (auto& future : fut_vec) {
        cols_entropies.insert(future.get());
    }*/

    col_idx_t col_selected = 0;
    double highest_ig = 0.0d;

    for (auto& ig : cols_entropies) {

        double col_entropy = 0.0d;

        for (auto& de : ig.second) {
            col_entropy += static_cast<double>(de.dataset->size()) / static_cast<double>(data->size()) * de.entropy;
        }

        double ig_col = attr_entropy - col_entropy;

        if (ig_col > highest_ig) {
            highest_ig  = ig_col;
            col_selected = ig.first;
        }
#ifndef NDEBUG
        cout << ">>> col idx: " << ig.first << ", ig=" << ig_col << endl;
#endif
    }

#ifndef NDEBUG
    cout << ">>> la columna con la ganancia mayor es: " << col_selected << ", ig=" << highest_ig << endl;
#endif

    std::vector<result_tree_t*> empty_vec;
    auto node_tree = new result_tree_t();
    node_tree->children =  empty_vec;
    node_tree->attr_name = col_names->at(col_selected);


    for (auto& ig : cols_entropies.at(col_selected)) {
        set<col_idx_t> local_ignored_cols(ignored_cols);

        local_ignored_cols.insert(col_selected);
        result_tree_t* nd = calculate_ig(ig.dataset, values, attr_col_idx, local_ignored_cols);
        nd->attr_value = ig.value;
        nd->root = node_tree;
        node_tree->children.push_back(nd);
    }

    for (auto& pair : cols_entropies) {
        for (auto& e : pair.second) {
            delete e.dataset;
        }
    }

    return node_tree;
}

double DecisionTree::calculate_dataset_entropy(ml::datacontainer_t& data, set<csv_field_t> values, col_idx_t attr_col_idx)
{
    std::map<csv_field_t, size_t> frequency_table;

    /*
     * Insertamos todos los valores posibles para la columna
     */
    for (auto& v : values) {
        frequency_table.insert(std::pair<csv_field_t, size_t>(v, 0));
    }

    /*
     * Calculamos la frecuencia de cada valor
     */
    for (std::vector<string>* s : *data) {
        frequency_table.at(s->at(attr_col_idx))++;
    }

    double entropy = 0.0d;

    /*
     * Calculo de la entropia
     */
    for (auto const& x : frequency_table) {
        double t =  static_cast<double>(x.second) /  static_cast<double>(data->size());
        double binary_log = 0.0d;

        /* evitamos calcular el logaritmo binario de cero */
        if (t > 0.0d) {
            binary_log = log2(t);
        }

        entropy -= t * binary_log;
    }

    return entropy;
}


std::pair<col_idx_t, std::vector<dataset_entropy>> DecisionTree::calculate_dataset_entropy_2_col(ml::datacontainer_t& data, set<string> values, col_idx_t attr_col_idx,  col_idx_t col)
{
    std::map<string, size_t> frequency_table;

    for (std::vector<string>* s : *data) {
        string col_value = s->at(col);

        if (frequency_table.count(col_value) == 0) {
            frequency_table.insert(std::pair<string, int>(col_value, 1));

        } else {
            frequency_table.at(col_value)++;
        }
    }

    //  double entropy = 0.0d;

    std::vector<dataset_entropy> dataset;

    for (auto const& x : frequency_table) {

        ml::datacontainer_t col_data;

        col_data = new  std::vector<std::vector<csv_field_t>*>();

        for (std::vector<string>* s : *data) {
            if (s->at(col) == x.first) {//s->at(col).compare(x.first) == 0 ) {
                col_data->push_back(s);
            }
        }

        double entropy_2col = calculate_dataset_entropy(col_data, values, attr_col_idx);

        dataset_entropy de = {entropy_2col, col_data, x.first};
        dataset.push_back(de);

        // entropy += static_cast<double>(col_data.size()) / static_cast<double>(data.size()) * entropy_2col;
    }

    return std::pair<col_idx_t, std::vector<dataset_entropy>>(col, dataset);
}

} /* namespace ml */
