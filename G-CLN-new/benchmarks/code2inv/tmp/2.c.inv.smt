(and (= (+ (- 30.0)
           (* (- 9.0) (* y y))
           (* 16.0 (* x x))
           (* (- 1.0) (* x y))
           (* 22.0 y)
           (* 42.0 x))
        0.0)
     (>= (* y y) 0.0)
     (>= (* x x) 1.0)
     (>= (* x y) 0.0)
     (>= y 0.0)
     (>= x 1.0))