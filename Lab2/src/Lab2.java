/**
 * CS838 Lab2
 * Team Member: Shuo Niu (sniu22@wisc.edu)
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

class Parser {
    public int[][] data;
    public ArrayList raw_protein = new ArrayList();

    public ArrayList samplex = new ArrayList();
    public ArrayList sampley = new ArrayList();

    public HashMap one_hot = new HashMap();

    public Parser(String filepath) {
        Scanner fileScanner = null;
        try {
            fileScanner = new Scanner(new File(filepath));
        } catch (FileNotFoundException e) {
            System.err.println("Could not find file '" + filepath +
                    "'.");
            System.exit(1);
        }
        boolean record = false;
        List protein = new ArrayList();
        int count = -1;
        int inputtype = 0;
        while (fileScanner.hasNextLine()) {
            String line = fileScanner.nextLine();
            if (line.equals("<end>") || line.equals("end") || line.equals("")) {
                continue;
            }
            if (line.equals("<>") && count == -1) {
                count += 1;
                record = true;
                continue;
            }
            if (line.equals("<>")) {
                count += 1;
                this.raw_protein.add(protein);
                protein = new ArrayList();
                continue;
            }
            if (record) {
                char input = line.charAt(0);
                if (!this.one_hot.containsKey(input)) {
                    this.one_hot.put(input, inputtype);
                    inputtype += 1;
                }
                protein.add(line);
            }
        }

        this.one_hot.put('0', 20);
    }

    public ArrayList[] protein_sample(ArrayList protein) {
        ArrayList x = new ArrayList();
        ArrayList y = new ArrayList();
        for (int i = 0; i < 8; i++) {
            protein.add(0, "00");
        }

        for (int i = 0; i < 8; i++) {
            protein.add(protein.size(), "00");
        }

        for (int i = 8; i < protein.size() - 8; i++) {
            int[] thisx = new int[17];
            String stry = (String) protein.get(i);
            if ('h' == stry.charAt(2)) {
                y.add(new int[]{1, 0, 0});
            } else if (stry.charAt(2) == 'e') {
                y.add(new int[]{0, 1, 0});
            } else {
                y.add(new int[]{0, 0, 1});
            }
            int position = 0;
            for (int j = i - 8; j < i + 8; j++) {
                String strx = (String) protein.get(j);
                thisx[position] = (Integer) this.one_hot.get(strx.charAt(0));
                position += 1;
            }

            int[][] sample = new int[17][this.one_hot.size()];
            for (int j = 0; j < 17; j++) {
                sample[j][thisx[j]] = 1;
            }
            x.add(sample);
        }
        return new ArrayList[]{x, y};
    }

    public void transfer() {
        ArrayList[] newp;
        for (int i = 0; i < this.raw_protein.size(); i++) {
            newp = this.protein_sample((ArrayList) this.raw_protein.get(i));
            this.samplex.add(newp[0]);
            this.sampley.add(newp[1]);
        }
    }

}


public class Lab2 {
    public double rate; public double decay; public double momentum; public int hidden_num;

    public double[][][] wei1; public double[][] wei2; public double[][][] ldelta_w1; public double[][] ldelta_w2;

    public double[] bias1; public double[] bias2; public double[] ldelta_b1; public double[] ldelta_b2;

    public double[] data; public String[] label; public int continue_epoch; public double dropout;

    public ArrayList train = new ArrayList();
    public ArrayList tune = new ArrayList();
    public ArrayList test = new ArrayList();

    public ArrayList trainC = new ArrayList();
    public ArrayList tuneC = new ArrayList();
    public ArrayList testC = new ArrayList();


    public void weight_initializer() {
        this.wei1 = new double[hidden_num][17][21]; this.ldelta_w1 = new double[hidden_num][17][21];

        this.wei2 = new double[3][hidden_num]; this.ldelta_w2 = new double[3][hidden_num];

        this.bias1 = new double[hidden_num]; this.ldelta_b1 = new double[hidden_num];

        this.bias2 = new double[3]; this.ldelta_b2 = new double[3];

        for (int i = 0; i < hidden_num ; i++) this.bias1[i] = (Math.random() - 0.5) / 1 ;

        for (int i = 0; i < 3 ; i++) this.bias2[i] = (Math.random() - 0.5) / 1 ;

        for (int i = 0; i < hidden_num; i++)
            for (int j = 0; j < 17; j++)
                for (int k = 0; k < 21 ; k++)
                    this.wei1[i][j][k] =  (Math.random() -0.5) / 1 ;

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < hidden_num; j++)
                this.wei2[i][j] =  (Math.random() - 0.5) / 1;
    }

    public Lab2(int hiddenNumber, Parser input,  double learn_rate, double decay_value,double momentum,
                int continue_epoch,double dropout) {
        this.decay = decay_value; this.rate = (-1)*learn_rate; this.momentum = momentum;
        this.hidden_num = hiddenNumber; this.weight_initializer(); this.continue_epoch = continue_epoch;
        this.dropout = dropout;
        int count = 0;
        for (int i = 0; i < input.samplex.size(); i++) {
            count += 1;
            ArrayList x = (ArrayList) input.samplex.get(i);
            ArrayList y = (ArrayList) input.sampley.get(i);
            for (int j = 0; j < x.size(); j++) {
                if (count % 5 == 0){
                    this.tune.add(x.get(j));
                    this.tuneC.add(y.get(j));
                } else if (count % 5 == 1 && count > 5) {
                    this.test.add(x.get(j));
                    this.testC.add(y.get(j));
                } else {
                    this.train.add(x.get(j));
                    this.trainC.add(y.get(j));
                }
            }
        }
    }

    public double[] hiddenout(int[][] sample) {
        double[] hiddenres = new double[hidden_num] ;
        for (int i = 0; i < hidden_num; i++) {
            hiddenres[i] = bias1[i];
            for (int j = 0; j < 17; j++)
                for (int k = 0; k < 21; k++)
                    hiddenres[i] += this.wei1[i][j][k]*sample[j][k];
        }
        //relu
        for (int i = 0; i < hidden_num; i++) {
            hiddenres[i] = Math.max(0,hiddenres[i]);
        }
        return hiddenres;
    }
    public int[] dropout(double dropsize){
        int[] drop = new int[this.hidden_num];
        for (int i = 0; i < this.hidden_num; i++) {
            if (Math.random() > dropsize) drop[i] = 1;
        }
        return drop;
    }

    public double[] finalout(double[] hiddenres, int[] dropout){
        double[] output = new double[3];
        for (int i = 0; i < 3; i++){
            output[i] = this.bias2[i];
            for (int j = 0; j < hidden_num; j++)
                output[i] += this.wei2[i][j]*hiddenres[j]*dropout[j];}
        //Sigmoid
        for (int i = 0; i < 3; i++) {
            output[i] = 1 / (1 + Math.exp(-1 * output[i]));
        }
        return output;
    }
    public void update(int[][] sample,int[] label,double[] hiddenres, double[] out,int[] dropout){
        double[][][] delta_w1 = new double[hidden_num][17][21];
        double[][] delta_w2 = new double[3][hidden_num];
        double[] delta_b1 = new double[hidden_num];
        double[] delta_b2 = new double[3];
        double[] part1 = new double[3];
        // calculate the gradient
        for (int i = 0; i < 3; i++) {
            part1[i] = (out[i] - (double)label[i])*Math.exp(-1*out[i])/Math.pow((1+Math.exp(-1*out[i])),2);
        }
//        System.out.println(" 1 : " + part1[0] + " 2 : " + part1[1] + " 3 : " + part1[2] );
        for (int i = 0; i < out.length; i++) {
            delta_b2[i] += part1[i];
            for (int j = 0; j < hidden_num; j++) {
                if (dropout[j] == 0) continue;

                delta_w2[i][j] += part1[i]*hiddenres[j];
                delta_b1[j] += part1[i]*this.wei2[i][j];
                for (int k = 0; k < 17; k++)
                    for (int l = 0; l < 21; l++)
                        delta_w1[j][k][l] += part1[i]*sample[k][l]*this.wei2[i][j];
            }
        }
        // calculate the delta, renew the last delta
        for (int i = 0; i < this.hidden_num ; i++) {
            if (dropout[i] == 0) continue;

            delta_b1[i] = (1-this.momentum)*(this.rate * delta_b1[i] + this.rate * this.decay * this.bias1[i]) + this.momentum *
                    this.ldelta_b1[i];
            this.ldelta_b1[i] = delta_b1[i];

            for (int j = 0; j < 17; j++) {
                for (int k = 0; k < 21; k++) {
                    delta_w1[i][j][k] = (1-this.momentum)*(this.rate*delta_w1[i][j][k] + this.rate * this.decay * this.wei1[i][j][k]) +
                            this.momentum * this.ldelta_w1[i][j][k];
                    this.ldelta_w1[i][j][k] = delta_w1[i][j][k];
                }
            }
        }
        for (int i = 0; i < 3; i++) {
            delta_b2[i] = (1-this.momentum)*(delta_b2[i]*this.rate + this.rate*this.decay*this.bias2[i]) + this.momentum*this.ldelta_b2[i];
            this.ldelta_b2[i] = delta_b2[i];
            for (int j = 0; j < this.hidden_num; j++) {
                if (dropout[j] == 0) continue;

                delta_w2[i][j] = (1-this.momentum)*(this.rate*delta_w2[i][j] + this.rate * this.decay * this.wei2[i][j]) +
                        this.momentum * this.ldelta_w2[i][j];
                this.ldelta_w2[i][j] = delta_w2[i][j];
            }
        }
        // Update the weight
        for (int i = 0; i < this.hidden_num ; i++) {
            if (dropout[i] == 0) continue;
            this.bias1[i] = this.bias1[i] + delta_b1[i];

            for (int j = 0; j < 17; j++)
                for (int k = 0; k < 21; k++)
                    this.wei1[i][j][k] = this.wei1[i][j][k] + delta_w1[i][j][k];
        }

        for (int i = 0; i < 3; i++) {
            this.bias2[i] = this.bias2[i] + delta_b2[i];
            for (int j = 0; j < this.hidden_num; j++) {
                if (dropout[j] == 0) continue;
                this.wei2[i][j] = this.wei2[i][j] + delta_w2[i][j];
            }
        }
    }

    public void one_epoch(){
        int[] dropout = this.dropout(this.dropout);
        for (int i = 0; i < this.train.size(); i++) {

            int[][] sample = (int[][]) this.train.get(i);
            int[] label = (int[]) this.trainC.get(i);
            double[] hiddenres = this.hiddenout(sample);
            double[] output = this.finalout(hiddenres,dropout);
            this.update(sample,label,hiddenres,output,dropout);
        }
    }

    public void train_model(){
        int decrease  = 3, epoch = 0;
        double last ,now = 0, best = 0;
        double[][][] b_wei1 = new double[hidden_num][17][21];
        double[][] b_wei2 = new double[3][hidden_num];
        double[] b_bias1 = new double[hidden_num]; double[] b_bias2 = new double[3];

        while(decrease > 0 && epoch < 80){
            last = now;
            this.one_epoch(); epoch += 1;
            if(epoch % 10 == 0) this.rate *= 0.2;
            now = this.accuracy(this.tune,this.tuneC,false);
            if (now > best){
                best = now;
                b_wei1 = this.wei1.clone(); b_wei2 = this.wei2.clone();
                b_bias1 = this.bias1.clone(); b_bias2 = this.bias2.clone();
            }
            if(last >= now) decrease -= 1;
            else decrease = 3;
//            System.out.println("train " + epoch + " " + this.accuracy(this.train,this.trainC,false));
//            System.out.println("tune " + epoch + " " + now);
//            System.out.println("test " + epoch + " " + this.accuracy(this.test,this.testC,false));
        }
        for (int i = 0; i < this.continue_epoch; i++) {
            this.one_epoch(); epoch += 1;

            if(epoch % 10 == 0) this.rate *= 0.2;
            now = this.accuracy(this.tune,this.tuneC,false);

            if (now > best){
                best = now;
                b_wei1 = this.wei1.clone(); b_wei2 = this.wei2.clone();
                b_bias1 = this.bias1.clone(); b_bias2 = this.bias2.clone();
            }
//            System.out.println("train " + epoch + " " + this.accuracy(this.train,this.trainC,false));
//            System.out.println("tune " + epoch + " " + now);
//            System.out.println("test " + epoch + " " + this.accuracy(this.test,this.testC,false));
        }
        this.wei1 = b_wei1; this.wei2 = b_wei2;
        this.bias1 = b_bias1; this.bias2 = b_bias2;
    }

    public double accuracy(ArrayList data, ArrayList label, boolean ifprint){

        String res = "";
        String[] slabel = {"h","e","_"};

        int[] dropout = new int[this.hidden_num];
        for (int i = 0; i < dropout.length; i++) dropout[i] = 1;

        double correct = 0;
        for (int i = 0; i < data.size(); i++) {
            int[][] sample = (int[][]) data.get(i);
            double[] hiddenres = this.hiddenout(sample);
            double[] out = this.finalout(hiddenres,dropout);
            int maxv = 0; double current = 0;
            for (int j = 0; j < out.length; j++) {
                if (out[j] > current) {
                    current = out[j];
                    maxv = j;
                }
            }
            int[] trueout = (int[])label.get(i);
            if (trueout[maxv] == 1) correct += 1;
            res += "Sample " + (i + 1) + " Prediction: " + slabel[maxv] + "\n";
        }
        res += "\nThe overall accuracy of test set is " + correct/data.size()*100 + "%\n";
        if (ifprint){
            System.out.println(res);
        }
        return correct/data.size();
    }


    public static void main(String[] args) {

        Parser data = new Parser(args[0]);
        data.transfer();

        Lab2 testmodel = new Lab2(50,data,0.05,0.0,0.9,
                10,0.5);
        testmodel.train_model();

        testmodel.accuracy(testmodel.test,testmodel.testC,true);
    }

}
