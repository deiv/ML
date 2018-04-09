//
// Created by deiv on 8/04/18.
//

#ifndef MLPRACTICALTEST_INPUTPARSER_H
#define MLPRACTICALTEST_INPUTPARSER_H

#include <string>
#include <vector>

#include "ml.h"

namespace ml {

class InputParser {
public:
    InputParser();
    ~InputParser();

    void parse_csv(std::string csv_path);
    datacontainer_t get_data() { return csv_data; }
    std::vector<csv_field_t>* get_col_names() { return csv_col_names; }

private:

    datacontainer_t csv_data;
    std::vector<csv_field_t>* csv_col_names;
};

} /* namespace ml */

#endif //MLPRACTICALTEST_INPUTPARSER_H
