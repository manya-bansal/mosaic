#ifndef INTERFACE_SCHEDULAR_H
#define INTERFACE_SCHEDULAR_H

#include <queue>

#include "taco/accelerator_notation/accelerator_notation.h"

class Pile {

    public:
        Pile(const taco::AcceleratorStmt& targetStmt) : targetStmt(targetStmt) {}
        taco::AcceleratorStmt getTargetStmt() const { return targetStmt; }

    private:
        taco::AcceleratorStmt targetStmt;
    // std::vector<Target> targets;


};

class Piles {
public:
    explicit Piles(std::function<bool(const Pile& p1, const Pile& p2)> cmpPriority) : pq(comp(cmpPriority)){}

    void feed(Pile x) { pq.push(x); }
    Pile get() { return pq.top(); }
    bool empty() {return pq.empty();}
    void pop() { pq.pop(); }

    struct comp {
        std::function<bool(const Pile& p1, const Pile& p2)> cmpPriority;
        comp(std::function<bool(const Pile& p1, const Pile& p2)> cmpPriority) : cmpPriority(cmpPriority) {}
        bool operator()(Pile a, Pile b) { return cmpPriority(a, b); };
    };

private:
    std::priority_queue<Pile, std::vector<Pile>, comp> pq;
};


#endif 