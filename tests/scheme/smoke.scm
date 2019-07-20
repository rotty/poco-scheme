;; Literals
(check "literal fixnum 1" 1 => 1)
(check "literal #t" #t => #t)
(check "literal #f" #f => #f)

;; Anonymouse procedure application
(check "0-arity closure application" ((lambda () 42)) => 42)
(check "unarity closure application" ((lambda (x) (+ x 23)) 42) => 65)
(check "binary closure application" ((lambda (x y) (+ x y)) 23 42) => 65)

(check "if expression in operrator position"
       ((lambda (mul? x y) ((if mul? * +) x y)) #f 2 3)
       => 5)

