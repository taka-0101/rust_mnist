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

fn transpose(v: Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    let mut result = Vec::new();
    for i in 0..v[0].len(){
        let mut r = Vec::new();
        for j in 0..v.len(){
            r.push(v[j][i])
        }
        result.push(r)
    }
    result
}

fn vector_dot(v1: Vec<Vec<f64>>, v2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    if v1[0].len() == v2.len() {
        for n in 0..v2[0].len() {
            let mut v = Vec::new();
            for i in 0..v1.len(){
                let mut value = 0.0;
                for j in 0..v2.len(){
                    value = value + v1[i][j] * v2[j][n]
                }
                v.push(value)
            }
            result.push(v)
        }
    }
    else{
        println!("ERROR: Not much vector size")
    }
    transpose(result)
}


fn main() {
    let ly = Layer::new(2, 3);
    println!("{}",ly.pre_neuron_size);
    println!("{}",ly.next_neuron_size);
    println!("{:?}", ly.weights);
    println!("{:?}", ly.biases);

    let v1 = vec![vec![1.0,2.0,3.0],vec![2.0,5.0,4.0]];
    let v2 = vec![vec![2.0,5.0],vec![3.0,4.0],vec![1.0,5.0]];
    println!("{:?}", vector_dot(v1,v2));

    let ly1 = Layer::new(4, 2);
    println!("{:?}", vector_dot(ly1.weights,ly.weights));


}
