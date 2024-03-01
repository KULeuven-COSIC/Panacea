use rand::seq::SliceRandom;
use std::collections::{hash_map::Entry, HashMap};
/// We just create a table for the digests
/// An improvement would be tabluation hashing, see https://en.wikipedia.org/wiki/Tabulation_hashing
#[derive(Clone, Debug)]
pub struct NaiveHash {
    tables: Vec<usize>,
    h_count: usize,
    input_domain: usize,
}

impl NaiveHash {
    pub fn new(h_count: usize, input_domain: usize) -> Self {
        assert!(h_count > 0);
        let mut rng = rand::thread_rng();
        let mut tables: Vec<usize> = (0..h_count * input_domain).collect();
        tables.shuffle(&mut rng);
        Self {
            tables,
            h_count,
            input_domain,
        }
    }

    pub fn checked_hash(&self, h: usize, x: usize) -> Option<usize> {
        if h >= self.h_count() || x >= self.input_domain() {
            None
        } else {
            Some(self.hash(h, x))
        }
    }

    pub fn hash(&self, h: usize, x: usize) -> usize {
        self.tables[self.h_count * x + h]
    }

    pub fn hash0(&self, x: usize) -> usize {
        self.hash(0, x)
    }

    pub fn hash1(&self, x: usize) -> usize {
        self.hash(1, x)
    }

    pub fn hash2(&self, x: usize) -> usize {
        self.hash(2, x)
    }

    pub const fn input_domain(&self) -> usize {
        self.input_domain
    }

    pub const fn h_count(&self) -> usize {
        self.h_count
    }

    pub const fn output_domain(&self) -> usize {
        self.input_domain() * self.h_count()
    }

    /// Hash the index `x` into a 2d point.
    pub fn hash_to_tuple(&self, h: usize, x: usize, d: usize) -> (usize, usize) {
        let tmp = self.hash(h, x);
        (tmp / d, tmp % d)
    }

    /// Find a mapping for indices in `is`.
    /// Elements in `is` should be unique.
    /// The hash map is keyed with the row index,
    /// the value is (column index, original index).
    pub fn hash_to_mapping(&self, is: &[usize], d: usize) -> HashMap<usize, (usize, usize)> {
        // TODO this is just a greedy algorithm, can be improved
        let mut out: HashMap<usize, (usize, usize)> = HashMap::new();
        for i in is {
            for h in 0..self.h_count() {
                let (r, c) = self.hash_to_tuple(h, *i, d);
                if let Entry::Vacant(e) = out.entry(r) {
                    e.insert((c, *i));
                    break;
                }
                if h >= self.h_count() {
                    panic!("all slots are taken for i={}", i);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod test {
    use super::NaiveHash;

    #[test]
    fn test_naive_hash() {
        let in_domain = 16usize;
        let nh = NaiveHash::new(1, in_domain);
        for i in 0..in_domain {
            assert!(nh.hash0(i) < in_domain);
        }

        assert_eq!(None, nh.checked_hash(1, 0));
        assert_eq!(None, nh.checked_hash(0, in_domain));
        let (r, c) = nh.hash_to_tuple(0, 8, 2);
        assert_eq!(nh.hash(0, 8), 2 * r + c);
    }

    #[test]
    fn test_naive_hash_to_mapping() {
        let in_domain = 16usize;
        let nh = NaiveHash::new(1, in_domain);
        let d = 2;
        let out = nh.hash_to_mapping(&[0], d);
        assert_eq!(out.len(), 1);

        let (r, c) = nh.hash_to_tuple(0, 0, 2);
        assert_eq!(out[&r].0, c);
    }
}
