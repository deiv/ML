
#include <iostream>
#include <sstream>
#include <string>

#include "../lib/rapidcsv.h"

#include "InputParser.h"
#include "ml.h"

using std::string;
using std::cout;
using std::endl;
using std::for_each;
using std::stringstream;

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

void print_result(ml::result_tree_t* result)
{
    if (result->root == nullptr && result->children.empty()) {
        cout << result->attr_value << endl;

    } else {
        for (ml::result_tree_t *t : result->children) {

            /* hoja */
            if (t->attr_name.empty()) {
                ml::result_tree_t *r = t;

                string line;

                while (r->root != nullptr) {
                    line.insert(0, " & " + r->root->attr_name + " = " + r->attr_value);
                    r = r->root;
                }
                cout << line.replace(0, 3, "") << endl;

            } else {
                print_result(t);
            }
        }
    }
}

void free_result(ml::result_tree_t* result)
{
    for (ml::result_tree_t* t : result->children) {

        /* hoja */
        if (t->attr_name.empty()) {
            delete t;

        } else {
            free_result(t);
        }
    }
}

int main(int argc, char **argv)
{
    struct program_args args = parse_arguments(argc, argv);

#ifndef NDEBUG
    cout
        << "csv de entrada: "
        << args.csv_path
        << ", indice del atributo a predecir: "
        << args.out_attr_idx
        << endl;
#endif

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

        ml::DecisionTree decisionTree(input_parser.get_data(), input_parser.get_col_names());
        ml::result_tree_t* result = decisionTree.create_decision_tree(args.out_attr_idx);

        print_result(result);
        free_result(result);

    } catch (io::error::can_not_open_file& ex) {
        std::cerr << "error: no se puede abrir el fichero: " << args.csv_path << endl;
        exit(EXIT_FAILURE);

    } catch (std::exception& ex) {
        std::cerr << "error inesperado " << ex.what() << endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
