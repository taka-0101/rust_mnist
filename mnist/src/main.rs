extern crate rand;
extern crate image;
use rand::Rng;
use std::fs;
use image::GenericImageView;
use image::DynamicImage;
use image::Rgba;

struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
}

impl Layer {
    fn new(p: u16, n: u16, b:u16) -> Layer {
        Layer{
            weights: Layer::weights_init(p, n),
            biases: Layer::biases_init(n, b)
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

    fn biases_init(n: u16, b:u16) -> Vec<Vec<f64>> {
        let mut biases = Vec::new();
        let mut rng = rand::thread_rng();
        for _j in 0..b{
            let mut b = Vec::new();
            for _i in 0..n {
                b.push(rng.gen::<f64>());
            }
            biases.push(b);
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
        println!("ERROR: Not much vector size")
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

fn cross_entropy_error(y:Vec<Vec<f64>>, t:Vec<Vec<f64>>) -> Vec<f64>{
    let delta = 1e-7;
    let mut result = Vec::new();
    let one = 1.0_f64;
    let e = one.exp();
    if y.len() == t.len(){
        for i in 0..y.len(){
            let mut r = 0.0 as f64;
            for j in 0..y[i].len(){
                let value = y[i][j] + delta;
                r = r + (t[i][j] * value.log(e));
            }
            result.push(r*-1.0);
        }
    }
    else{
        println!("ERROR:Vector not mutch size, cross_entropy_error");
    }
    result
}

fn identity_function(x: Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    let mut result = Vec::new();
    for i in 0..x.len(){
        let mut value = 0.0 as f64;
        let mut r = Vec::new();
        for j in 0..x[i].len(){
            value = value + x[i][j];
        }
        for j in 0..x[i].len(){
            r.push(x[i][j] / value);
        }
        result.push(r);
    }
    result
}

fn load_image(path: &str) -> Vec<f64>{
    let img: DynamicImage = image::open(path).unwrap();
    let (width, height) = img.dimensions();

    let mut result = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let pixel: Rgba<u8> = img.get_pixel(x, y);
            if pixel[0] > 0 {
                result.push(1.0 as f64);
            }
            else {
                result.push(0.0 as f64);
            }
        }
    }
    result
}


fn read_file(path: &str) -> Vec<Vec<f64>> {
    let paths = fs::read_dir(path).unwrap();
    let mut result = Vec::new();
    for (i, p) in paths.enumerate() {
        //println!("Name: {}", p.unwrap().path().display())
        let p = p.unwrap().path();
        let p_ = p.to_string_lossy().to_string();
        result.push(load_image(&p_));
        if i == 99{
            break;
        }
    }
    result
}

fn main() {
    let ly1 = Layer::new(784, 50, 100);
    let ly2 = Layer::new(50, 50, 100);
    let ly3 = Layer::new(50, 10, 100);
    
    let n: Vec<f64> = vec![2.5, 3.0];
    let mut neu = Vec::new();
    neu.push(n);
    let neu = read_file("C:/Users/rakun/Documents/mnist_png/mnist_png/testing/0/");
    
    let mut label = Vec::new();
    
    for _i in 0..100 {
        let mut kari = Vec::new();
        kari.push(1.0 as f64);
        for _i in 0..9 {
            kari.push(0.0 as f64)
        }
        label.push(kari)
    }

    let r1 = calc_neuron(neu, ly1);
    let r2 = calc_neuron(sigmoid(r1), ly2);
    let r3 = calc_neuron(sigmoid(r2), ly3);
    let result = cross_entropy_error(identity_function(r3), label);
    println!("{:?}", result);

}