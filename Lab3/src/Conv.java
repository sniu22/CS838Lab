import java.util.ArrayList;
import java.util.List;

/**
 * Created by Lament on 2017/2/28.
 */
public class Conv {
    public double[][][][] weight,w_delta,wm_delta,wv_delta;
    public double[] bias,b_delta,bm_delta,bv_delta;

    public ArrayList<double[][][][]> batch_w_gradient = new ArrayList<>();
    public ArrayList<double[]> batch_b_gradient = new ArrayList<>();


    public int inchannel,outchannel, side, width, mini_batch, update_count = 0;
    public double leaky,learn_rate, beta_1 = 0.9, beta_2 = 0.999;

    /**
     * @param side
     *              conv size, i.e. 3*3
     * @param width
     *              image size, i.e. 32*32
     * @param outchannel
     *              number of output channels
     * @param inchannel
     *              number of input channels
     * @param leaky
     *              coefficient of leaky relu activation
     */
    public Conv(int side, int width, int outchannel, int inchannel,double leaky, double learn_rate, int mini_batch){
        this.side = side; this.inchannel = inchannel; this.outchannel = outchannel; this.width = width;
        this.leaky = leaky; this.learn_rate = learn_rate;
        this.bias = new double[outchannel];
        this.b_delta = new double[outchannel];
        this.bm_delta = new double[outchannel];
        this.bv_delta = new double[outchannel];

        this.weight = new double[outchannel][inchannel][side][side];
        this.wm_delta = new double[outchannel][inchannel][side][side];
        this.wv_delta = new double[outchannel][inchannel][side][side];

        this.w_delta = new double[outchannel][inchannel][side][side];
        this.mini_batch = mini_batch;
        this.weight_initializer();
    }

    public void weight_initializer() {
        for (int i = 0; i < outchannel; i++)
            for (int j = 0; j < inchannel; j++)
                for (int k = 0; k < side; k++)
                    for (int l = 0; l < side; l++)
                        // weight initialized between 0 and 1
                        this.weight[i][j][k][l] = (Tool.random()-0.5)/Math.sqrt(this.width*this.width*this.inchannel);
//        for (int i = 0; i < outchannel; i++) {
//            // bias initialized between 0 and 0.1
//            bias[i] = Lab3.randomNormal(0.001);
//        }

    }

    /**
     * @param input
     *              the original input 3-d array
     * @return 3-d array after padding
     */
    public double[][][] padding(double[][][] input){
        int pad_size = this.side/2;
        double[][][] new_input = new double[this.inchannel][this.width + 2*pad_size][this.width + 2*pad_size];
        for (int channel = 0; channel < this.inchannel; channel++) {
            for (int i = 0; i < this.width; i++) {
                for (int j = 0; j <this.width; j++) {
                    new_input[channel][i + pad_size ][j + pad_size] = input[channel][i][j];
                }
            }
        }
        return new_input;
    }

    /**
     * @param pad
     *              input 3-d array after padding
     * @return 3-d array removed padding elements
     */

    public double[][][] de_pad(double[][][] pad){
        int pad_size = this.side/2;
        int p_channel = pad.length;
        int p_width = pad[0].length - 2*pad_size;

        double[][][] res = new double[p_channel][p_width][p_width];

        for (int i = 0; i < p_channel; i++) {
            for (int j = 0; j < p_width; j++) {
                for (int k = 0; k < p_width; k++) {
                    res[i][j][k] = pad[i][j+pad_size][k+pad_size];
                }
            }
        }
        return res;
    }
    /**
     * @param input
     *              the original input 3-d array
     * @return 3-d output of this conv layer
     */

    public double[][][] compute(double[][][] input){
        double[][][] out = new double[this.outchannel][this.width][this.width];
        double[][][] pad_data = this.padding(input);

        // The conv result

        for (int outc = 0; outc < this.outchannel; outc++)
            for (int channel = 0; channel < this.inchannel; channel++)
                for (int i = 0; i < this.width; i++)
                    for (int j = 0; j < this.width; j++)
                        for (int w = 0; w < this.side; w++)
                            for (int h = 0; h < this.side; h++)
                                out[outc][i][j] += this.weight[outc][channel][w][h] * pad_data[channel][i + w][j + h];

        // Add the bias item and activation function
        for (int i = 0; i < this.outchannel; i++) {
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < width; k++) {
                    // Add the bias value
                    out[i][j][k] += this.bias[i];
                    // Leaky relu function
                    if (out[i][j][k] < 0){
                        out[i][j][k] *= this.leaky;
                    }
                }
            }
        }
        return out;
    }


    public void update(double[][][] part1, double[][][] input, double[][][] output){
        double[][][] pad_input = this.padding(input);
        double[][][][] w_gradient = new double[outchannel][inchannel][side][side];
        double[] b_gradient= new double[outchannel];

        for (int outc = 0; outc < this.outchannel; outc++)
            for (int channel = 0; channel < this.inchannel; channel++)
                for (int i = 0; i < this.width; i++)
                    for (int j = 0; j < this.width; j++){
                        double lambda = (output[outc][i][j] > 0)? 1:leaky ;
                        b_gradient[outc] += lambda * part1[outc][i][j];
                        for (int w = 0; w < this.side; w++)
                            for (int h = 0; h < this.side; h++)
                               w_gradient[outc][channel][w][h] += lambda * part1[outc][i][j] * pad_input[channel][i + w][j + h];}
        this.batch_w_gradient.add(w_gradient);
        this.batch_b_gradient.add(b_gradient);

        if (this.batch_w_gradient.size() == this.mini_batch) {
            this.update_count += 1;
            // calculate the mean gradient
            double[][][][] w_mean_gradient = new double[outchannel][inchannel][side][side];
            double[] b_mean_gradient= new double[outchannel];
            for (int i = 0; i < this.batch_w_gradient.size(); i++) {
                w_mean_gradient = Tool.mat_4d(w_mean_gradient,this.batch_w_gradient.get(i),1,1/(double)this.mini_batch,"add");
                b_mean_gradient = Tool.mat_1d(b_mean_gradient,this.batch_b_gradient.get(i),1,1/(double)this.mini_batch,"add");
            }
            this.batch_w_gradient = new ArrayList<>();
            this.batch_b_gradient = new ArrayList<>();

            //adam delta for weight
            wm_delta = Tool.mat_4d(wm_delta,w_mean_gradient,beta_1,1 - beta_1 ,"add");
            wv_delta = Tool.mat_4d(wv_delta,Tool.mat_4d(w_mean_gradient,null,2,1,"constant_pow"),
                    beta_2,1-beta_2,"add");

            //adam delta for bias
            bm_delta = Tool.mat_1d(bm_delta,b_mean_gradient,beta_1,1 - beta_1  ,"add");
            bv_delta = Tool.mat_1d(bv_delta,Tool.mat_1d(b_mean_gradient,null,2,1,"constant_pow"),
                    beta_2,1-beta_2,"add");

            // update the bias and weight
            for (int oc = 0; oc < this.outchannel; oc++) {
                bias[oc] = bias[oc] - this.learn_rate/(Math.sqrt(bv_delta[oc]/(1-Math.pow(beta_2,update_count))) + 10e-8)
                        *bm_delta[oc]/(1-Math.pow(beta_1,update_count));
                for (int ic = 0; ic < this.inchannel; ic++) {
                    for (int i = 0; i < this.side; i++) {
                        for (int j = 0; j < this.side; j++) {
                            weight[oc][ic][i][j] = weight[oc][ic][i][j] - this.learn_rate/
                                    (Math.sqrt(wv_delta[oc][ic][i][j]/(1-Math.pow(beta_2,update_count))) + 10e-8)
                                    *wm_delta[oc][ic][i][j]/(1-Math.pow(beta_1,update_count));
                        }
                    }
                }
            }
        }



//            // Update the weight by SGD
//
//            for (int oc = 0; oc < this.outchannel; oc++) {
//                double b_update = -this.learn_rate * b_gradient[oc] + this.momentum * this.b_delta[oc];
//                b_delta[oc] = b_update;
//                bias[oc] += b_update;
//                for (int ic = 0; ic < this.inchannel; ic++) {
//                    for (int i = 0; i < this.side; i++) {
//                        for (int j = 0; j < this.side; j++) {
//                            double w_update = -this.learn_rate * w_gradient[oc][ic][i][j] +
//                                    this.momentum * this.w_delta[oc][ic][i][j];
//                            w_delta[oc][ic][i][j] = w_update;
//                            weight[oc][ic][i][j] += w_update;
//                        }
//                    }
//                }
//            }

            // Update the weight by Adam
            // calculate the update delta

    }

    /**
     * Back Propagation to previous layers
     * @param part1
     *              gradient backprop from other layers
     * @param output
     *                output of this layer
     * @return 3-d array filled gradient passed to previous conv layer
     */

    public double[][][] c_back(double[][][] part1, double[][][] output){

        double[][][] pad_res = padding(new double[this.inchannel][this.width][this.width]);
        for (int outc = 0; outc < this.outchannel; outc++)
            for (int channel = 0; channel < this.inchannel; channel++)
                for (int i = 0; i < this.width; i++)
                    for (int j = 0; j < this.width; j++){
                        double lambda = (output[outc][i][j] > 0)? 1 : leaky ;
                        for (int w = 0; w < this.side; w++)
                            for (int h = 0; h < this.side; h++)
                                pad_res[channel][i + w][j + h] += this.weight[outc][channel][w][h] * lambda * part1[outc][i][j];
        }
        return this.de_pad(pad_res);
    }
}

