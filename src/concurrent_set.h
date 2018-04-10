//
// Created by david.suarez on 10/04/2018.
//

#ifndef MLPRACTICALTEST_CONCURRENT_SET_H
#define MLPRACTICALTEST_CONCURRENT_SET_H

#include <cstddef>
#include <set>
#include <mutex>

class FlagContainer {
public:
    FlagContainer()
    {
        array.fill(false);
    }

    void setFlagOn(size_t n)
    {
       array[n] = true;
    }

    void setFlagOff(size_t n)
    {
        array[n] = false;
    }

    void set_flag(size_t n, bool v)
    {
        array[n] = v;
    }

    bool get_flag(size_t n)
    {
        return array[n];
    }

    ~FlagContainer()
    {
       // delete array;
    }

private:
    std::array<bool, 20> array;
};

template <typename T, typename Compare = std::less<T>>
    class concurrent_set
    {
    private:
        std::set<T, Compare> set_;
        mutable std::mutex mutex_;

    public:
        typedef typename std::set<T, Compare>::iterator iterator;

        concurrent_set() {};
        concurrent_set( concurrent_set<T, Compare>& __x)
          {
            std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
            set_.insert(__x.set_.begin(), __x.set_.end());
        }

        std::pair<iterator, bool>
        insert(const T& val) {
            std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
            return set_.insert(val);
        }

        size_t size() const {
            std::unique_lock<std::mutex> lock3(mutex_, std::defer_lock);
            return set_.size();
        }

        size_t count(T element) const {
            std::lock_guard<std::mutex> lg(mutex_);
            return set_.count(element);
        }

        std::set<T, Compare> getSet() { std::lock_guard<std::mutex> lg(mutex_); return set_; }
    };


#endif //MLPRACTICALTEST_CONCURRENT_SET_H
