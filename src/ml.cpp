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

const char* FAILED_ATTR_NAME  = "Failure";
const char* FAILED_ATTR_VALUE = "No input_data";
const char* EMPTY_STRING      = "";
const double ENTROPY_ZERO = 0.0;

result_tree_t* create_tree_node(result_tree_t* parent_node, string attr_name, string attr_value)
{
    auto node = new result_tree_t();

    node->root = parent_node;
    node->attr_name = attr_name;
    node->attr_value = attr_value;

    return node;
}

result_tree_t* DecisionTree::create_decision_tree(col_idx_t out_attr_idx)
{
    set<csv_field_t> out_attr_values;

    /*
     * Obtenemos los valores únicos para la columna indice.
     */
    for (auto& s : *input_data) {
        out_attr_values.insert(s->at(out_attr_idx));
    }

    /*
      * Si la lista de atributos esta vacia, devolvemos el valor más frecuente del
      * atributo de salida.
      */
    if (input_col_names->empty()) {
        string most_frequent_value = get_most_frequent_value(input_data, out_attr_values, out_attr_idx);
        return create_tree_node(nullptr, EMPTY_STRING, most_frequent_value);
    }

    set<col_idx_t> empty_set;

    return calculate_tree_node(input_data, out_attr_values, out_attr_idx, empty_set);
}

result_tree_t* DecisionTree::calculate_tree_node(datacontainer_t &data, set<csv_field_t> out_attr_values, col_idx_t out_attr_idx, set<col_idx_t>& ignored_cols)
{
/*
     * Si no existen datos, devolvemos un nodo indicando el error.
     */
    if (data->empty()) {
        return create_tree_node(nullptr, FAILED_ATTR_NAME, FAILED_ATTR_VALUE);
    }

    /*
     * Calculamos la entropia de la columna a predecir
     */
    double attr_entropy = calculate_dataset_entropy(data, out_attr_values, out_attr_idx);

    /*
     * caso uno: ''If all the output values are the same in dataset, return a leaf node that says
     *             “predict this unique output”''
     */
    if (attr_entropy == ENTROPY_ZERO) {
       /* auto node      = create_tree_node(nullptr, input_col_names->at(out_attr_idx), "predict this unique output");
        auto leaf_node = create_tree_node(node, "" , data->at(0)->at(out_attr_idx));

        node->children.push_back(leaf_node);*/

        return create_tree_node(nullptr, "", "predict this unique output");
    }

    /*
     * Calculamos la ganancia de las columnas
     */
    return calculate_ig(data, out_attr_values, out_attr_idx, ignored_cols, attr_entropy);
}

/*
 * IG(Y% X) = H(Y) − H(Y% X)
 */
result_tree_t* DecisionTree::calculate_ig(datacontainer_t &data, set<csv_field_t> values, col_idx_t attr_col_idx, set<col_idx_t>& ignored_cols, double out_attr_entropy)
{
    size_t col_count = data->at(0)->size();
    map<col_idx_t, vector<dataset_entropy>> cols_entropies;
    //vector<future<pair<col_idx_t, vector<dataset_entropy>>>> cols_entropies_fut_vec;

    /*
     * Lanzamos el calculo de las entropias de cada columna en paralelo
     */
    for (col_idx_t col_idx = 0; col_idx < col_count; col_idx++) {

        /* no contamos con la columna de salida ... */
        if (col_idx == attr_col_idx) {
            continue;
        }

        /* ... ni con las columnas ya partidas. */
        if (ignored_cols.count(col_idx) != 0) {
            continue;
        }


        cols_entropies.insert(calculate_dataset_entropy_2_col(data, values, attr_col_idx, col_idx));
    }

    col_idx_t col_selected = 0;
    double highest_ig = 0.0;

    for (auto& ig : cols_entropies) {

        double col_entropy = 0.0;

        for (auto& de : ig.second) {
            col_entropy += static_cast<double>(de.dataset->size()) / static_cast<double>(data->size()) * de.entropy;
        }

        double ig_col = out_attr_entropy - col_entropy;

        if (ig_col > highest_ig) {
            highest_ig  = ig_col;
            col_selected = ig.first;
        }
    }

    result_tree_t* node_tree;

    /*
     * caso 2: ''Don’t split a node if none of the attributes can create multiple nonempty children
     */
    if (highest_ig == 0.0 ) {

        node_tree = create_tree_node(nullptr, "", "predict the majority output");
    } else {
#ifndef NDEBUG
        cout << ">>> la columna con la ganancia mayor es: " << col_selected << ", ig=" << highest_ig << endl;
#endif

        node_tree = new result_tree_t();
        node_tree->attr_name = input_col_names->at(col_selected);

        for (auto &ig : cols_entropies.at(col_selected)) {
            set<col_idx_t> local_ignored_cols(ignored_cols);
            local_ignored_cols.insert(col_selected);

            result_tree_t *nd = calculate_tree_node(ig.dataset, values, attr_col_idx, local_ignored_cols);

            if (nd != nullptr) {
                nd->attr_value = ig.value;
                nd->root = node_tree;
                node_tree->children.push_back(nd);
            }
        }
    }

    for (auto& pair : cols_entropies) {
        for (auto& e : pair.second) {
            delete e.dataset;
        }
    }

    return node_tree;
}

std::map<csv_field_t, size_t> DecisionTree::get_frequency_table(ml::datacontainer_t& data, set<csv_field_t> values, col_idx_t attr_col_idx)
{
    std::map<csv_field_t, size_t> frequency_table;

    /*
     * Insertamos todos los valores posibles para la columna
     */
    for (auto& value : values) {
        frequency_table.insert(std::pair<csv_field_t, size_t>(value, 0));
    }

    /*
     * Calculamos la frecuencia de cada valor
     */
    for (std::vector<string>* s : *data) {
        frequency_table.at(s->at(attr_col_idx))++;
    }

    return frequency_table;
}

string DecisionTree::get_most_frequent_value(datacontainer_t& data, set<csv_field_t> values, col_idx_t attr_col_idx)
{
    std::map<csv_field_t, size_t> frequency_table = get_frequency_table(data, values, attr_col_idx);

    size_t max_frequency = 0;
    csv_field_t field_selected;

    for (auto const& x : frequency_table) {
        if (x.second > max_frequency) {
            max_frequency = x.second;
            field_selected = x.first;
        }
    }

    return field_selected;
}


double DecisionTree::calculate_dataset_entropy(ml::datacontainer_t& data, set<csv_field_t> values, col_idx_t attr_col_idx)
{
    std::map<csv_field_t, size_t> frequency_table = get_frequency_table(data, values, attr_col_idx);

    double entropy = 0.0;

    /*
     * Calculo de la entropia
     */
    for (auto const& x : frequency_table) {
        double t =  static_cast<double>(x.second) /  static_cast<double>(data->size());
        double binary_log = 0.0;

        /* evitamos calcular el logaritmo binario de cero */
        if (t > 0.0) {
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
    }

    return std::pair<col_idx_t, std::vector<dataset_entropy>>(col, dataset);
}

} /* namespace ml */
