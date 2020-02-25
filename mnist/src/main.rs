extern crate rand;
extern crate image;
use rand::Rng;
use image::GenericImageView;
use image::DynamicImage;
use image::Rgba;

struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
}

impl Layer {
    fn new(p: u16, n: u16) -> Layer {
        Layer{
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

    fn biases_init(n: u16) -> Vec<Vec<f64>> {
        let mut biases = Vec::new();
        let mut b = Vec::new();
        let mut rng = rand::thread_rng();
        for _i in 0..n {
            b.push(rng.gen::<f64>());
        }
        biases.push(b);
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

fn vector_sum(v1: Vec<Vec<f64>>, v2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    if v1.len() == v2.len() && v1[0].len() == v2[0].len() {
        for i in 0..v1.len(){
            let mut r = Vec::new();
            for j in 0..v1[i].len(){
                r.push(v1[i][j] + v2[i][j])
            }
            result.push(r)
        }
    }
    result
}

fn calc_neuron(neuron: Vec<Vec<f64>>, ly: Layer) -> Vec<Vec<f64>> {
    let r = vector_dot(neuron, ly.weights);
    let result = vector_sum(r, ly.biases);
    result
}

fn sigmoid(x: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    for i in 0..x.len(){
        let mut r = Vec::new();
        for j in 0..x[i].len(){
            r.push(1.0 / (1.0 + f64::exp(-x[i][j])));
        }
        result.push(r);
    }
    result
}

fn identity_function(x: Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    x
}

fn load_image(path: &str) -> Vec<Vec<f64>>{
    let img: DynamicImage = image::open(path).unwrap();
    let (width, height) = img.dimensions();

    let mut result = Vec::new();
    let mut r = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let pixel: Rgba<u8> = img.get_pixel(x, y);
            if pixel[0] > 0 {
                r.push(1.0 as f64);
            }
            else {
                r.push(0.0 as f64);
            }
        }
    }
    result.push(r);
    result
}

fn main() {
    let ly1 = Layer::new(784, 50);
    let ly2 = Layer::new(50, 50);
    let ly3 = Layer::new(50, 2);
    
    let n: Vec<f64> = vec![2.5, 3.0];
    let mut neu = Vec::new();
    neu.push(n);

    let r1 = calc_neuron(load_image("mnist_png/testing/0/3.png"), ly1);
    let r2 = calc_neuron(sigmoid(r1), ly2);
    let r3 = calc_neuron(sigmoid(r2), ly3);

    println!("{:?}", identity_function(r3));

}