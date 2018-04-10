//
// Created by deiv on 8/04/18.
//

#ifndef MLPRACTICALTEST_ML_H
#define MLPRACTICALTEST_ML_H

#include <cstddef>
#include <vector>
#include <set>
#include <map>

namespace ml {

/*
 * Definición de tipos
 */
typedef std::string csv_field_t;
typedef std::vector<std::vector<csv_field_t>*>* datacontainer_t;
typedef size_t col_idx_t;
typedef double entropy_t;

struct dataset_entropy {
    entropy_t entropy;
    datacontainer_t dataset;
    csv_field_t value;
};

typedef struct result_tree_t {
    result_tree_t* root;
    csv_field_t attr_name;
    csv_field_t attr_value;
    std::vector<result_tree_t*> children;
} result_tree_t ;

class DecisionTree {
public:
    DecisionTree(ml::datacontainer_t data, std::vector<csv_field_t>* col_names)
        : input_data(data),
          input_col_names(col_names) {};

    result_tree_t* create_decision_tree(col_idx_t out_attr_idx);

private:

    result_tree_t* calculate_tree_node(datacontainer_t &data, std::set<csv_field_t> out_attr_values, col_idx_t out_attr_idx, std::set<col_idx_t>& ignored_cols);
    result_tree_t* calculate_ig(datacontainer_t &data, std::set<csv_field_t> values, col_idx_t attr_col_idx, std::set<col_idx_t>& ignored_cols, double out_attr_entropy);
    std::map<csv_field_t, size_t> get_frequency_table(ml::datacontainer_t& data, std::set<csv_field_t> values, col_idx_t attr_col_idx);
    std::string get_most_frequent_value(datacontainer_t& data, std::set<csv_field_t> values, col_idx_t attr_col_idx);
    double calculate_dataset_entropy(ml::datacontainer_t& data, std::set<csv_field_t> values, col_idx_t attr_col_idx);
    std::pair<col_idx_t, std::vector<dataset_entropy>> calculate_dataset_entropy_2_col(ml::datacontainer_t& data, std::set<csv_field_t> values, col_idx_t attr_col_idx,  col_idx_t col);

    datacontainer_t input_data;
    std::vector<csv_field_t>* input_col_names;
};

} /* namespace ml */

#endif //MLPRACTICALTEST_ML_H
