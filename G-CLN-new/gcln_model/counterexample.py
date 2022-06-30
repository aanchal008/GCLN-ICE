test1 = 'checking ../benchmarks/code2inv/smt2/501.c:\n(let ((a!1 (and (>= (+ 36.0 (* 45.0 x) (* (- 40.0) y)) 0.0)))) (or a!1))\nsmt check for pre-condition:\n(error "line 64 column 0: invalid command, \'(\' expected")\nunsat\n(error "line 72 column 10: model is not available")\nsmt check for recursive condition\nunsat\n(error "line 73 column 10: model is not available")\nsmt check for post-condition\nsat\n(\n  (define-fun y_2 () Int\n    1001)\n  (define-fun x_2 () Int\n    1000)\n  (define-fun y () Int\n    1001)\n  (define-fun x () Int\n    1000)\n  (define-fun x_1 () Int\n    0)\n  (define-fun x_3 () Int\n    0)\n  (define-fun y_0 () Int\n    0)\n  (define-fun y_3 () Int\n    0)\n  (define-fun x! () Int\n    0)\n  (define-fun y! () Int\n    0)\n  (define-fun x_0 () Int\n    0)\n  (define-fun y_1 () Int\n    0)\n)\n'
test2 = 'checking ../benchmarks/code2inv/smt2/501.c:\n(let ((a!1 (and (>= (+ 36.0 (* 45.0 x) (* (- 40.0) y)) 0.0) (< x 1000.0)))) (or a!1))\nsmt check for pre-condition:\n(error "line 64 column 0: invalid command, \'(\' expected")\nunsat\n(error "line 72 column 10: model is not available")\nsmt check for recursive condition\nsat\n(\n  (define-fun x () Int\n    999)\n  (define-fun x! () Int\n    1000)\n  (define-fun y! () Int\n    1)\n  (define-fun y () Int\n    1)\n  (define-fun x_2 () Int\n    0)\n  (define-fun x_1 () Int\n    0)\n  (define-fun x_3 () Int\n    0)\n  (define-fun y_0 () Int\n    0)\n  (define-fun y_2 () Int\n    0)\n  (define-fun y_3 () Int\n    0)\n  (define-fun x_0 () Int\n    0)\n  (define-fun y_1 () Int\n    0)\n)\nsmt check for post-condition\nunsat\n(error "line 71 column 10: model is not available")\n'
stdout_string =  test2
stdout_lines = stdout_string.split('\n')
counter_example = { "pre" : {},
                    "recursive" : {},
                    "post" : {}
                  }
index = ""
for i in range(len(stdout_lines)):
    line = stdout_lines[i]
    if "pre-condition" in line:
        index = "pre"
    if "recursive condition" in line:
        index = "recursive"
    if "post-condition" in line:
        index = "post"
    if "define-fun" in line:
        strings = line.split()
        variable_name = strings[1]
        line = stdout_lines[i+1]
        strings = line.split()
        variable_value = strings[0].rstrip(')')

        counter_example[index][variable_name] = variable_value
        #print("\n")

print(counter_example)
print(counter_example["recursive"]['x'])
#to check if pre is unsatisfied
if not counter_example["pre"]:
    print("No counter example found for pre")
else:
    print("Counter examples found for pre")