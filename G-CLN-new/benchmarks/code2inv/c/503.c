int main() {
  // variable declarations
  int x;
  int y;
  // pre-conditions
  assume((y<x));
  (x = 0);
  (y = 0);
  // loop body
  while ((x < 100 )) {
    {
      if ((y % 2) == 0)
      {
        x = x + 1;
      }
    (y  = (y + 1));
    }
  }
  // post-condition
assert( (2*y >= x) );
}

//503.b assert(y <= 8x)