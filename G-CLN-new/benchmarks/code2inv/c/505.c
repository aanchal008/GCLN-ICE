int main() {
  // variable declarations
  int x;
  // pre-conditions
  assume((0<x));
  (x = 0);
  // loop body
  while ((x < 100)) {
    {
    (x  = (x - 1));
    }

  // post-condition
assert( (x <= 100) );
}
}

