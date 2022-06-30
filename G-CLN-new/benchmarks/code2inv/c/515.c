int main() {
  // variable declarations
  int x;
  int y;
  // pre-conditions
  assume((y<x));
  (x = 0);
  (y = 0);
  // loop body
  while ((x < 100)) {
    {
    (x  = (x + 3));
    (y  = (y + 1));
    }

  }
  // post-condition
assert( (x >= y) );
}

