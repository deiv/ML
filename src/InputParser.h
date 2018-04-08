//
// Created by deiv on 8/04/18.
//

#ifndef MLPRACTICALTEST_INPUTPARSER_H
#define MLPRACTICALTEST_INPUTPARSER_H

#include <string>
#include <vector>

namespace ml {

typedef std::vector<std::vector<std::string>> datacontainer_t;

class InputParser {
public:
    InputParser() {};

    void parse_csv(std::string csv_path);

    /* XXX: typedef */
    datacontainer_t& get_data() { return csv_data; }

private:
    datacontainer_t csv_data;
};

} /* namespace ml */

#endif //MLPRACTICALTEST_INPUTPARSER_H
