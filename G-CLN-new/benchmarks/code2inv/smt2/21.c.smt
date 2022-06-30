(set-logic LIA)

( declare-const m Int )
( declare-const m! Int )
( declare-const n Int )
( declare-const n! Int )
( declare-const x Int )
( declare-const x! Int )
( declare-const tmp Int )
( declare-const tmp! Int )

( declare-const m_0 Int )
( declare-const m_1 Int )
( declare-const m_2 Int )
( declare-const m_3 Int )
( declare-const n_0 Int )
( declare-const x_0 Int )
( declare-const x_1 Int )
( declare-const x_2 Int )

( define-fun inv-f( ( m Int )( n Int )( x Int )( tmp Int ) ) Bool
SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop
)

( define-fun pre-f ( ( m Int )( n Int )( x Int )( tmp Int )( m_0 Int )( m_1 Int )( m_2 Int )( m_3 Int )( n_0 Int )( x_0 Int )( x_1 Int )( x_2 Int ) ) Bool
	( and
		( >= n_0 2 )
		( = x 0 )
		( = m 0 )
	)
)

( define-fun trans-f ( ( m Int )( n Int )( x Int )( tmp Int )( m! Int )( n! Int )( x! Int )( tmp! Int )( m_0 Int )( m_1 Int )( m_2 Int )( m_3 Int )( n_0 Int )( x_0 Int )( x_1 Int )( x_2 Int ) ) Bool
	( or
		( and
			( = m_1 m )
			( = x_1 x )
			( = m_1 m! )
			( = x_1 x! )
			( = n n_0 )
			( = n! n_0 )
			( = m m! )
			(= tmp tmp! )
		)
		(and 
		    (< x 200 )
		    (=> true (= m! x)) 
			(=> (distinct (mod x 2) 0) (= m! m)) 
			(= x! (+ x 1))
		)
	)
)

( define-fun post-f ( ( m Int )( n Int )( x Int )( tmp Int )( m_0 Int )( m_1 Int )( m_2 Int )( m_3 Int )( n_0 Int )( x_0 Int )( x_1 Int )( x_2 Int ) ) Bool
	( or
		( not
			( and
				( = m m_1)
				( = n n_0 )
				( = x x_1)
			)
		)
		( not
			( and
				( not ( < x_1 200 ) )
				( not ( >= (* 2 x_1) m_1 ) ) 
			)
		)
	)
)
SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop
( assert ( not
	( =>
		( pre-f m n x tmp m_0 m_1 m_2 m_3 n_0 x_0 x_1 x_2  )
		( inv-f m n x tmp )
	)
))

SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop
( assert ( not
	( =>
		( and
			( inv-f m n x tmp )
			( trans-f m n x tmp m! n! x! tmp! m_0 m_1 m_2 m_3 n_0 x_0 x_1 x_2 )
		)
		( inv-f m! n! x! tmp! )
	)
))

SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop
( assert ( not
	( =>
		( inv-f m n x tmp  )
		( post-f m n x tmp m_0 m_1 m_2 m_3 n_0 x_0 x_1 x_2 )
	)
))

