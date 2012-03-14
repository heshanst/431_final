#include "MatrixUtil.hpp"
#include <cassert>
#include <iostream>

using namespace std;

void test() {
  Matrix m1(2,2);
  m1[0][0] = 1;
  m1[0][1] = 2;
  m1[1][0] = 3;
  m1[1][1] = 4;

  Matrix m2(2,2);
  m2[0][0] = 4;
  m2[0][1] = 3;
  m2[1][0] = 2;
  m2[1][1] = 1;
  
  Matrix m3(2,2,5);
  assert (m1+m2 == m3);

  m1 += m2;
  assert(m1 == m3);
}

int main() {
  test();
}
