int main() {
  // variable declarations
  int x;
  // pre-conditions
  assume((0<x));
  (x = 101);
  // loop body
  while ((x > 0)) {
    {
    (x  = (x - 1));
    }

  // post-condition
assert( (x >= 0) );
}
}

                                                                             