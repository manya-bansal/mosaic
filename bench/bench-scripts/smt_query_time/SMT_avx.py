import z3
import timeit

start = timeit.default_timer()

s = z3.Solver()
i2302 = z3.Int('i2302')
s.add(i2302 > 0)
s.add(i2302 < 10)

s.add(z3.And((i2302) == (4)))



while s.check() == z3.sat:
	# print(s.model())
	s.add(z3.Or(i2302 > s.model()[i2302]))

end = timeit.default_timer()

print(f"Time taken to query for AVX is {end-start}s.")
