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

void InputParser::parse_csv(std::string csv_path)
{
    io::LineReader in(csv_path);
    char* line_data;

    /* saltamos el encabezado csv */
    in.next_line();

    while (line_data = in.next_line()) {

        string line(line_data);
        vector<string> line_data;
        size_t last = 0;
        size_t next = 0;

        while ((next = line.find(";", last)) != string::npos) {
            line_data.push_back(line.substr(last, next - last));
            last = next + 1;
        }

        line_data.push_back(line.substr(last));

        csv_data.push_back(line_data);
    }
}

} /* namespace ml */
