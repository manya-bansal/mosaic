import z3
import timeit

start = timeit.default_timer()

s = z3.Solver()
i2302 = z3.Int('i2302')
s.add(i2302 > 0)
s.add(i2302 < 20)
i2303 = z3.Int('i2303')
s.add(i2303 > 0)
s.add(i2303 < 20)
i2304 = z3.Int('i2304')
s.add(i2304 > 0)
s.add(i2304 < 20)
i2305 = z3.Int('i2305')
s.add(i2305 > 0)
s.add(i2305 < 20)

s.add((i2302 * i2303 * i2304 * i2305) == (65536))

while s.check() == z3.sat:
	s.add(z3.Or(i2302 > s.model()[i2302], i2303 > s.model()[i2303], i2304 > s.model()[i2304], i2305 > s.model()[i2305]))

end = timeit.default_timer()

print(f"Time taken to query for Stardust is {end-start}s.")
