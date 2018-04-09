//
// Created by deiv on 8/04/18.
//

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../lib/rapidcsv.h"

#include "InputParser.h"

using std::string;
using std::cout;
using std::endl;
using std::vector;

namespace ml {

const char* CSV_SEPARATOR = ";";

std::vector<csv_field_t>* cut_line_cols( const string& line)
{
    auto container = new std::vector<csv_field_t>();
    size_t last = 0;
    size_t next = 0;

    while ((next = line.find(CSV_SEPARATOR, last)) != string::npos) {
        container->push_back(line.substr(last, next - last));
        last = next + 1;
    }

    container->push_back(line.substr(last));

    return container;
}

InputParser::InputParser()
{
    csv_data = new std::vector<std::vector<csv_field_t>*>();
    csv_col_names = new std::vector<csv_field_t>();
};

InputParser::~InputParser()
{
    delete csv_col_names;
    delete csv_data;
};

void InputParser::parse_csv(std::string csv_path)
{
    io::LineReader in(csv_path);
    char* line_data;

    line_data = in.next_line();
    csv_col_names = cut_line_cols(line_data);

    while ((line_data = in.next_line())) {
        csv_data->push_back(cut_line_cols(line_data));
    }
}

} /* namespace ml */
