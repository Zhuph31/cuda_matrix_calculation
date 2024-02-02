#include <iostream>
#include <unistd.h>

int main(int argc, char *argv[]) {

  std::string row_str, col_str;
  int row, col;

  if ((argc <= 2)) {
    std::cerr << "Error: not enough input for row and column number.\n";
    return -1;
  } else if (argc > 3) {
    std::cerr << "Error: too many input arguments.\n";
    return -1;
  } else {
    row_str = argv[1];
    col_str = argv[2];

    try {
      row = std::stoi(row_str);
      col = std::stoi(col_str);
    } catch (std::exception &e) {
      std::cerr << "Error, failed to convert row/col number to integer, error "
                   "message:"
                << e.what() << '\n';
      return -1;
    }
  }

  printf("specified row:%d, col:%d\n", row, col);

  return 0;
}