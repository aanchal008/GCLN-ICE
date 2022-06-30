int main() {
  // variable declarations
  int x;
  int y;
  // pre-conditions
  assume((y >= x));   
  (x = 0);
  (y = 0);
  
  while ((x < 100)) {
    {
      x = x + 1;
      y = y + 2;
    }

  }
  // post-condition
  assert( (y >= x) );
}

//502.b assert (2y >= x) 

