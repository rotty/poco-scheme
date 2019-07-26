;; Literals
(check "literal fixnum 1" 1 => 1)
(check "literal #t" #t => #t)
(check "literal #f" #f => #f)

;; Begin sanity check
(check "begin returns last expression" (begin 1 2 3) => 3)

;; Anonymouse procedure application
(check "0-arity closure application" ((lambda () 42)) => 42)
(check "unarity closure application" ((lambda (x) (+ x 23)) 42) => 65)
(check "binary closure application" ((lambda (x y) (+ x y)) 23 42) => 65)

(check "if expression in operator position"
       ((lambda (mul? x y) ((if mul? * +) x y)) #f 2 3)
       => 5)

(check "definitions inside procedure"
       ((lambda ()
          (define foo 1)
          (define bar foo)
          (+ bar foo)))
       => 2)

(check "modulo" (modulo 42 9) => 6)

;; `define` and recursive bindings
(check "tail-recursive sum"
       (begin
         (define (sum n acc)
           (if (< n 1) acc (sum (- n 1) (+ acc n))))
         (sum 100 0))
       => 5050)
(check "closure with captured binding"
       (begin
         (define (make-adder n)
           (lambda (x) (+ x n)))
         (define add-5 (make-adder 5))
         (add-5 23))
       => 28)
(check "define procedure with reference to later definition"
       (begin
         (define (foo x) (+ x bar))
         (define bar 23)
         (foo 42))
       => 65)
(check-fail "detect undefined identifier in definition body"
            (define (foo) bar))
