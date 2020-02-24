extern crate rand;
use rand::Rng;

struct Layer {
    pre_neuron_size: u16,
    next_neuron_size: u16,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

impl Layer {
    fn new(p: u16, n: u16) -> Layer {
        Layer{
            pre_neuron_size: p,
            next_neuron_size: n,
            weights: Layer::weights_init(p, n),
            biases: Layer::biases_init(n)
        }
        
        
    }

    fn weights_init(p: u16, n: u16) -> Vec<Vec<f64>> {
        let mut weights = Vec::new();
        let mut rng = rand::thread_rng();
        for _i in 0..p {
            let mut w = Vec::new();
            for _j in 0..n {
                w.push(rng.gen::<f64>());
            }
            weights.push(w);
        }
        weights
    }

    fn biases_init(n: u16) -> Vec<f64> {
        let mut biases = Vec::new();
        let mut rng = rand::thread_rng();
        for _i in 0..n {
            biases.push(rng.gen::<f64>());
        }
        biases
    }
}

fn main() {
    let ly = Layer::new(2, 3);
    println!("{}",ly.pre_neuron_size);
    println!("{}",ly.next_neuron_size);
    println!("{:?}", ly.weights);
    println!("{:?}", ly.biases);
}
