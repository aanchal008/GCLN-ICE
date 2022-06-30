int main() {
  // variable declarations
  int x;
  int y;
  // pre-conditions
  (x = 100);
  (y = 0);
  // loop body
  while ((x > 0)) {
    {
    (y  = (y + 1));
    (x  = (x - 1));
    }

  }
  // post-condition
assert( (x + y <= 100) );
}
