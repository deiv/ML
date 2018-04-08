
#include <iostream>
#include <sstream>
#include <string>

#include <map>
#include <set>

#include <cmath>

#include <numeric>
#include <future>


#include "../lib/rapidcsv.h"

#include "InputParser.h"

using std::string;
using std::cout;
using std::endl;
using std::for_each;
using std::stringstream;
using std::set;

struct program_args {
    string csv_path;
    size_t out_attr_idx;
};

struct program_args parse_arguments(int argc, char **argv)
{
    /*
     * Comprobamos que tenemos 2 argumentos de entrada
     */
    if (argc != 3) {
        cout << "Número de parametros erroneo, uso: "
            << argv[0]
            << " input-file output-attribute-index"
            << endl;

        exit(EXIT_FAILURE);
    }

    struct program_args args = {};

    stringstream csv_path_arg;
    csv_path_arg << argv[1];
    csv_path_arg >> args.csv_path;

    /* XXX: comprobar entero */
    stringstream out_attr_idx_arg;
    out_attr_idx_arg << argv[2];
    out_attr_idx_arg >> args.out_attr_idx;

    return args;
}

double calculate_dataset_entropy(ml::datacontainer_t& data, set<string> values, size_t attr_col_idx)
{
    std::map<string, size_t> frequency_table;

    for (auto& v : values) {
        frequency_table.insert(std::pair<string, size_t>(v, 0));
    }

    /*
     * cogemos los valores de la columna
     */
    for (std::vector<string> s : data) {
       frequency_table.at(s.at(attr_col_idx))++;
    }

    double entropy = 0.0d;

    for (auto const& x : frequency_table) {
        double t =  static_cast<double>(x.second) /  static_cast<double>(data.size());
        double binary_log = 0.0d;

        if (t > 0.0d) {
            binary_log = log2(t);
        }

        entropy -= t * binary_log;
    }

    return entropy;
}

typedef size_t col_idx_t;
typedef double entropy_t;

struct dataset_entropy {
    entropy_t entropy;
    ml::datacontainer_t dataset;
};

std::pair<col_idx_t, std::vector<dataset_entropy>> calculate_dataset_entropy_2_col(ml::datacontainer_t& data, set<string> values, col_idx_t attr_col_idx,  col_idx_t col)
{
    std::map<string, size_t> frequency_table;

    for (std::vector<string> s : data) {
        string col_value = s.at(col);

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

        for (std::vector<string> s : data) {
            if (s.at(col).compare(x.first) == 0 ) {
                col_data.push_back(s);
            }
        }

        double entropy_2col = calculate_dataset_entropy(col_data, values, attr_col_idx);

        dataset_entropy de = {entropy_2col, col_data};
        dataset.push_back(de);

       // entropy += static_cast<double>(col_data.size()) / static_cast<double>(data.size()) * entropy_2col;
    }

    return std::pair<col_idx_t, std::vector<dataset_entropy>>(col, dataset);
}

int find_ig_attr2(ml::datacontainer_t& data, size_t attr_col_idx, set<string> values) {


    double attr_entropy = calculate_dataset_entropy(data, values, attr_col_idx);

    if (attr_entropy == 0.0d) {
        std::cout << "---" << endl;
        std::cout << "attr_entropy == 0" << endl;
        for (auto const& x : data)
        {
            std::cout << x.at(attr_col_idx)  // string (key)
                      << std::endl ;
        }
        std::cout << "---" << endl;
        return 0;
    }
    size_t col_number = data.at(0).size();
    std::map<col_idx_t, std::vector<dataset_entropy>> cols_entropies;
    std::vector<std::future<std::pair<col_idx_t, std::vector<dataset_entropy>>>> fut_vec;

    for (size_t col_idx = 0; col_idx < col_number; col_idx++) {
        if (col_idx == attr_col_idx) {
            break;
        }

        fut_vec.push_back(
                std::async(
                        std::launch::async,
                        calculate_dataset_entropy_2_col,
                        std::ref(data),
                        values,
                        attr_col_idx,
                        col_idx));
    }

    for (auto &future : fut_vec) {
        cols_entropies.insert(future.get());
    }

    col_idx_t col_selected = -1;
    double highest_ig = 0.0d;

    for (auto &ig : cols_entropies) {

        double col_entropy = 0.0d;

        for (auto &de : ig.second) {
            col_entropy += static_cast<double>(de.dataset.size()) / static_cast<double>(data.size()) * de.entropy;
        }

        double ig_col = attr_entropy - col_entropy;

        if (ig_col > highest_ig) {
            highest_ig = ig_col;
            col_selected = ig.first;
        }
    }

    std::cout << "---" << endl;
    std::cout << highest_ig << endl;
    std::cout << col_selected << endl;
    std::cout << "---" << endl;

    for (auto& ig : cols_entropies.at(col_selected)) {
        find_ig_attr2(ig.dataset, attr_col_idx, values);
    }
}


/*
 * IG(Y% X) = H(Y) − H(Y% X)
 */
int find_ig_attr(ml::datacontainer_t& data, size_t attr_col_idx)
{
    set<string> values;

    for (auto& s : data) {
        values.insert(s.at(attr_col_idx));
    }

    double attr_entropy = calculate_dataset_entropy(data, values, attr_col_idx);

    size_t col_number = data.at(0).size();
    std::map<col_idx_t, std::vector<dataset_entropy>> cols_entropies;
    std::vector<std::future<std::pair<col_idx_t, std::vector<dataset_entropy>>>> fut_vec;

    for (size_t col_idx = 0; col_idx < col_number; col_idx++) {
        if (col_idx == attr_col_idx) {
            break;
        }

        fut_vec.push_back(
            std::async(
                std::launch::async,
                calculate_dataset_entropy_2_col,
                std::ref(data),
                values,
                attr_col_idx,
                col_idx));
    }

    for (auto& future : fut_vec) {
        cols_entropies.insert(future.get());
    }

    col_idx_t col_selected = -1;
    double highest_ig = 0.0d;

    for (auto& ig : cols_entropies) {

        double col_entropy = 0.0d;

        for (auto& de : ig.second) {
            col_entropy += static_cast<double>(de.dataset.size()) / static_cast<double>(data.size()) * de.entropy;
        }

        double ig_col = attr_entropy - col_entropy;

        if (ig_col > highest_ig) {
            highest_ig  = ig_col;
            col_selected = ig.first;
        }
    }

    std::cout << "---" << endl;
    std::cout << highest_ig << endl;
    std::cout << col_selected << endl;
    std::cout << "---" << endl;

    for (auto& ig : cols_entropies.at(col_selected)) {
        find_ig_attr2(ig.dataset, attr_col_idx, values);
    }
}

int main(int argc, char **argv)
{
    struct program_args args = parse_arguments(argc, argv);

    cout
        << "csv de entrada: "
        << args.csv_path
        << ", indice del atributo a predecir: "
        << args.out_attr_idx
        << endl;

    try {
        ml::InputParser input_parser;

        input_parser.parse_csv(args.csv_path);


/*#ifndef NDEBUG
        for_each(
            input_parser.get_data().begin(),
            input_parser.get_data().end(),
            [](std::vector<string>& line_data)  {
                cout << "---" << endl;
                for_each(
                    line_data.begin(),
                    line_data.end(),
                    [](string str)  {
                        cout << str << endl;
                    }
                );
            }
        );
#endif*/

        find_ig_attr(input_parser.get_data(), args.out_attr_idx);

    } catch (io::error::can_not_open_file ex) {
        std::cerr << "error: no se puede abrir el fichero: " << args.csv_path << endl;
        exit(EXIT_FAILURE);

    } catch (std::exception ex) {
        std::cerr << "error inesperado " << ex.what() << endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
