(define (prime? n)
  (define bound (sqrt n))
  (define (loop divisor)
    (if (> divisor bound)
        #t
        (if (= 0 (modulo n divisor))
            #f
            (loop (+ divisor 2)))))
  (loop 3))

(define (show-prime n)
  (display n)
  (newline))

(define (list-primes bound)
  (define (loop n)
    (if (prime? n)
        (show-prime n))
    (if (< n bound)
        (loop (+ n 2))))
  (show-prime 2)
  (loop 3))

(list-primes 1000000)
