extern crate rand;
extern crate image;
use rand::Rng;
use std::fs;
use image::GenericImageView;
use image::DynamicImage;
use image::Rgba;
use std::io::{Write, Read, BufWriter, BufReader, copy};
use std::error::Error;
use std::str::FromStr;
use std::io::BufRead;

struct Dataset{
    image_patch: Vec<Vec<Vec<f64>>>,
    label_patch: Vec<Vec<Vec<f64>>>,
}

struct TwoLayerNet {
    w1 : Vec<Vec<f64>>,
    w2 : Vec<Vec<f64>>,
    b1 : Vec<f64>,
    b2 : Vec<f64>,
}

impl TwoLayerNet {
    fn new(input_size: u16, hidden_size: u16, output_size:u16) -> TwoLayerNet {
        let mut w1_ = TwoLayerNet::create_queue(input_size, hidden_size);
        let mut b1_ = TwoLayerNet::create_vector(hidden_size);
        let mut w2_ = TwoLayerNet::create_queue(hidden_size, output_size);
        let mut b2_ = TwoLayerNet::create_vector(output_size);
        
        TwoLayerNet{
            w1 : w1_,
            b1 : b1_,
            w2 : w2_,
            b2 : b2_,
        }
    }

    fn new_read_weight(path: &str, input_size: u16, hidden_size: u16, output_size:u16) -> TwoLayerNet {
        let mut w1_ = TwoLayerNet::create_queue(input_size, hidden_size);
        let mut b1_ = TwoLayerNet::create_vector(hidden_size);
        let mut w2_ = TwoLayerNet::create_queue(hidden_size, output_size);
        let mut b2_ = TwoLayerNet::create_vector(output_size);

        let weight_data = TwoLayerNet::read_weight_data(&(path.to_owned() + "weight1.txt"));
        if ( w1_.len() * w1_[0].len() ) == weight_data.len(){
            for i in 0..w1_.len(){
                for j in 0..w1_[i].len(){
                    w1_[i][j] = weight_data[(i*w1_[i].len()+j)];
                }
            }

        }
        else{
            println!("weight_size {}, weight_data_size{}", w1_.len() * w1_[0].len(), weight_data.len());
            println!("error: not match size new_read_weight1");
        }

        let weight_data = TwoLayerNet::read_weight_data(&(path.to_owned() + "weight2.txt"));
        if ( w2_.len() * w2_[0].len() ) == weight_data.len(){
            for i in 0..w2_.len(){
                for j in 0..w2_[i].len(){
                    w2_[i][j] = weight_data[(i*w2_[i].len()+j)];
                }
            }

        }
        else{
            println!("weight_size {}, weight_data_size{}", w2_.len() * w2_[0].len(), weight_data.len());
            println!("error: not match size new_read_weight2");
        }

        let biase_data = TwoLayerNet::read_weight_data(&(path.to_owned() + "biase1.txt"));
        if b1_.len() == biase_data.len(){
            for i in 0..b1_.len(){
                b1_[i] = biase_data[i];
            }
        }
        else{
            println!("biase_size {}, biase_data_size{}", b1_.len(), biase_data.len());
            println!("error: not match size new_read_biase1");
        }

        let biase_data = TwoLayerNet::read_weight_data(&(path.to_owned() + "biase2.txt"));
        if b2_.len() == biase_data.len(){
            for i in 0..b2_.len(){
                b2_[i] = biase_data[i];
            }
        }
        else{
            println!("biase_size {}, biase_data_size{}", b2_.len(), biase_data.len());
            println!("error: not match size new_read_biase2");
        }
        

        TwoLayerNet{
            w1 : w1_,
            b1 : b1_,
            w2 : w2_,
            b2 : b2_,
        }
    }

    fn write_weight_data(&self, path: &str){
        let mut f = fs::File::create(&(path.to_owned() + "weight1.txt")).unwrap(); 
        for i in 0..self.w1.len(){
            for j in 0..self.w1[i].len(){
                let string = self.w1[i][j].to_string() + "\n";
                f.write_all(string.as_bytes()).unwrap();
            }
        }
        f = fs::File::create(&(path.to_owned() + "weight2.txt")).unwrap(); 
        for i in 0..self.w2.len(){
            for j in 0..self.w2[i].len(){
                let string = self.w2[i][j].to_string() + "\n";
                f.write_all(string.as_bytes()).unwrap();
            }
        }
        f = fs::File::create(&(path.to_owned() + "biase1.txt")).unwrap(); 
        for i in 0..self.b1.len(){
            let string = self.b1[i].to_string() + "\n";
            f.write_all(string.as_bytes()).unwrap();
        }
        f = fs::File::create(&(path.to_owned() + "biase2.txt")).unwrap(); 
        for i in 0..self.b2.len(){
            let string = self.b2[i].to_string() + "\n";
            f.write_all(string.as_bytes()).unwrap();
        }

    }

    fn read_weight_data(path: &str) -> Vec<f64>{
        let f = match fs::File::open(&path) {
            Err(why) => panic!("couldn't open weight_file"),
            Ok(file) => file,
        };

        let reader = BufReader::new(f);
        let mut data = Vec::new();


        for line in reader.lines() 
        {
            let f_s = f64::from_str(&line.unwrap()).unwrap();    //リテラル（&str）をfloatに変換
            data.push(f_s);

        }
        data

    }

    fn create_queue(height: u16, wight: u16) -> Vec<Vec<f64>> {
        let mut result = Vec::new();
        let mut rng = rand::thread_rng();
        for _i in 0..height {
            let mut r = Vec::new();
            for _j in 0..wight {
                r.push(rng.gen_range::<f64>(-1.0, 1.0));
            }
            result.push(r);
        }
        result
    }

    fn create_vector(wight: u16) -> Vec<f64> {
        let mut result = Vec::new();
        let mut rng = rand::thread_rng();
        for _i in 0..wight {
            result.push(rng.gen_range::<f64>(-1.0, 1.0));
        }
        result
    }

    fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
        let r_1 = vector_dot(x, &self.w1);
        let r1 = vector_sum(&r_1, &self.b1);
        let r_2 = vector_dot(&sigmoid(&r1), &self.w2);
        let r2 = vector_sum(&r_2, &self.b2);
        let y = softmax(&r2);

        transpose(y)
    }

    fn loss(&self, x: &Vec<Vec<f64>>, t: &Vec<Vec<f64>>) -> f64{
        let y = self.predict(x);
        let result = cross_entropy_error(&y, t);
        result
    }


    fn learn(&mut self, x: &Vec<Vec<f64>>, t: &Vec<Vec<f64>>, learning_rate:f64) -> f64{
        let h = 0.0001;
        let mut w1_grads = Vec::new();
        println!("Gradient calculation");
        for i in 0..self.w1.len(){
            let mut grad = Vec::new();
            for j in 0..self.w1[i].len(){
                let value = self.w1[i][j];
                let value_high = value + h;
                let value_low = value - h;

                self.w1[i][j] = value_high;
                let fxh1 = self.loss(x, t);
                
                self.w1[i][j] = value_low;
                let fxh2 = self.loss(x, t);
                grad.push((fxh1 - fxh2) / (2.0 * h));

                self.w1[i][j] = value;
            }
            w1_grads.push(grad);
            println!("w1 calc gradient {} / {}", i, self.w1.len());
        }
        println!("w1 finished");
        //println!("{:?}", w1_grads);
        let mut b1_grads = Vec::new();
        for i in 0..self.b1.len(){
            let value = self.b1[i];
            let value_high = value + h;
            let value_low = value - h;

            self.b1[i] = value_high;
            let fxh1 = self.loss(x, t);
            
            self.b1[i] = value_low;
            let fxh2 = self.loss(x, t);

            self.b1[i] = value;
            b1_grads.push((fxh1 - fxh2) / (2.0 * h));
            println!("b1 calc gradient {} / {}", i, self.b1.len());
        }
        println!("b1 finished");
        //println!("{:?}", b1_grads);
        let mut w2_grads = Vec::new();
        for i in 0..self.w2.len(){
            let mut grad = Vec::new();
            for j in 0..self.w2[i].len(){
                let value = self.w2[i][j];
                let value_high = value + h;
                let value_low = value - h;

                self.w2[i][j] = value_high;
                let fxh1 = self.loss(x, t);
                
                self.w2[i][j] = value_low;
                let fxh2 = self.loss(x, t);
                grad.push((fxh1 - fxh2) / (2.0 * h));

                self.w2[i][j] = value;
            }
            w2_grads.push(grad);
            println!("w2 calc gradient {} / {}", i, self.w2.len());
        }
        println!("w2 finished");
        //println!("{:?}", w2_grads);
        let mut b2_grads = Vec::new();
        for i in 0..self.b2.len(){
            let value = self.b2[i];
            let value_high = value + h;
            let value_low = value - h;

            self.b2[i] = value_high;
            let fxh1 = self.loss(x, t);
            
            self.b2[i] = value_low;
            let fxh2 = self.loss(x, t);

            self.b2[i] = value;
            b2_grads.push((fxh1 - fxh2) / (2.0 * h));
            println!("b2 calc gradient {} / {}", i, self.b2.len());
        }
        println!("b2 finished");
        //println!("{:?}", b2_grads);

        println!("start : Update of paramater");
        for i in 0..self.w1.len(){
            for j in 0..self.w1[i].len(){
                self.w1[i][j] -= learning_rate * w1_grads[i][j];
            }
        }

        for i in 0..self.b1.len(){
            self.b1[i] -= learning_rate * b1_grads[i];

        }

        for i in 0..self.w2.len(){
            for j in 0..self.w2[i].len(){
                self.w2[i][j] -= learning_rate * w2_grads[i][j];
            }
        }

        for i in 0..self.b2.len(){
            self.b2[i] -= learning_rate * b2_grads[i];
        }
        println!("finish : Update of paramater");
        let result = self.loss(x, t);
        result
    }
}

fn softmax(v: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    let mut result = Vec::new();
    for i in 0..v[0].len(){
        let mut value = 0.0 as f64;
        let mut r = Vec::new();
        for j in 0..v.len(){
            value = value + f64::exp(v[j][i]);
        }
        
        for j in 0..v.len(){
            r.push(f64::exp(v[j][i]) / value);
        }
        result.push(r);
    }
    result
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

fn vector_dot(v1: &Vec<Vec<f64>>, v2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    if v1[0].len() == v2.len() {
        for i in 0..v1.len(){
            let mut r = Vec::new();
            for j in 0..v2[0].len(){
                let mut value = 0.0 as f64;
                for n in 0..v1[i].len(){
                    value += (v1[i][n] * v2[n][j]);
                }
                r.push(value);
            }
            result.push(r);
        }
    }
    else{
        println!("ERROR: Not much vector size");
    }
    result
}

fn vector_sum(v1: &Vec<Vec<f64>>, v2: &Vec<f64>) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    if v1[0].len() == v2.len(){
        for i in 0..v1.len(){
            let mut r = Vec::new();
            for j in 0..v1[i].len(){
                r.push(v1[i][j] + v2[i])
            }
            result.push(r)
        }
    }
    else{
        println!("ERROR not match size vector_sum");
    }
    result
}

fn sigmoid(x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

fn cross_entropy_error(y:&Vec<Vec<f64>>, t:&Vec<Vec<f64>>) -> f64{
    let delta = 1e-7;
    let one = 1.0_f64;
    let e = one.exp();
    let mut r = 0.0 as f64;
    if y.len() == t.len(){
        for i in 0..y.len(){
            for j in 0..y[i].len(){
                let value = y[i][j] + delta;
                r = r + (t[i][j] * value.log(e));
            }
        }
        
    }
    else{
        println!("ERROR:Vector not mutch size, cross_entropy_error");
    }
    
    let result = -1.0 * r / (y.len() as f64);
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

fn read_file(path: &str, patch_size: u16) -> Dataset {
    println!("start : road dataset");
    let mut rng = rand::thread_rng();
    let mut images_list = Vec::new();
    let Slash: String = "/".to_string();
    for num in 0..10{
        let file_path = path.to_owned() + "/" + &num.to_string();
        let paths = fs::read_dir(&file_path).unwrap();
        let mut images = Vec::new();
        for (i, p) in paths.enumerate() {
            let p = p.unwrap().path();
            let p_ = p.to_string_lossy().to_string();
            images.push(load_image(&p_));
            if i == 500{
                break;
            }
        }
        images_list.push(images);
    }

    let mut patchs = Vec::new();
    let mut labels_list = Vec::new();
    for num in 0..5000/patch_size{
        let mut patch = Vec::new();
        let mut labels = Vec::new();
        let mut answer = 0;
        let mut count = 0;
        for i in 0..patch_size{
            let mut label = Vec::new();
            let mut image = Vec::new();
            for c in 0..images_list[answer][count].len(){
                image.push(images_list[answer][count][c])
            }
            patch.push(image);
            
            for l in 0..10{
                if l == answer{
                    label.push(1.0 as f64);
                }
                else{
                    label.push(0.0 as f64);
                }
            }
            labels.push(label);

        }
        if answer > 10{
            answer = 0;
            count += 1;
        }
        patchs.push(patch);
        labels_list.push(labels);
    }
    let result = Dataset{image_patch: patchs, label_patch: labels_list};
    
    println!("finish : road dataset");
    result
    
}

fn main() {
    let dataset_ = read_file("C:/Users/rakun/Documents/mnist_png/mnist_png/testing/", 10);

    //let mut Net = TwoLayerNet::new_read_weight("C:/Users/rakun/Documents/rust_mnist/mnist/result/0_loss_0/",784,30,10);
    let mut Net = TwoLayerNet::new(784,30,10);
    
    let mut loss = Vec::new();
    for i in 0..dataset_.image_patch.len(){
        let loss_ = Net.learn(&dataset_.image_patch[i], &dataset_.label_patch[i], 0.3);
        let result_path = "result/".to_owned() + &i.to_string() + "_loss_" + &loss_.to_string() + "/";
        fs::create_dir_all(&result_path);
        Net.write_weight_data(&result_path);
        loss.push(loss_);
        println!("loss {:?}", loss);
    }


}