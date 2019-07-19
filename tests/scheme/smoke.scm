(check 1 => 1)
(check #t => #t)
(check #f => #f)

(check ((lambda () 42)) => 42)
(check ((lambda (x) (+ x 23)) 42) => 65)
