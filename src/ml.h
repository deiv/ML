//
// Created by deiv on 8/04/18.
//

#ifndef MLPRACTICALTEST_ML_H
#define MLPRACTICALTEST_ML_H

#include <cstddef>
#include <vector>
#include <set>

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
    csv_field_t attr_name;                       /* null == hoja */
    csv_field_t attr_value;                      /* null == raiz */
    std::vector<result_tree_t*> children;
} result_tree_t ;

class DecisionTree {
public:
    DecisionTree(ml::datacontainer_t data, std::vector<csv_field_t>* col_names)
        : data(data),
          col_names(col_names) {};

    result_tree_t* create_decision_tree(col_idx_t attr_col_idx);

private:

    result_tree_t* calculate_ig(datacontainer_t &data, std::set<csv_field_t> values, col_idx_t attr_col_idx, std::set<col_idx_t>& ignored_cols);
    double calculate_dataset_entropy(ml::datacontainer_t& data, std::set<csv_field_t> values, col_idx_t attr_col_idx);
    std::pair<col_idx_t, std::vector<dataset_entropy>> calculate_dataset_entropy_2_col(ml::datacontainer_t& data, std::set<std::string> values, col_idx_t attr_col_idx,  col_idx_t col);

    datacontainer_t data;
    std::vector<csv_field_t>* col_names;
};

} /* namespace ml */

#endif //MLPRACTICALTEST_ML_H
