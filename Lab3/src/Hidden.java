import java.util.ArrayList;

public class Hidden{
    private int h_count, input_size,mini_batch, update_count = 0;
    public double[] dropout;
    public double dropoutsize, leaky, learn_rate, beta_1 = 0.9, beta_2 = 0.999;


    public ArrayList<double[][]> batch_w_gradient = new ArrayList<>();
    public ArrayList<double[]> batch_b_gradient = new ArrayList<>();

    public double[] bias,b_delta,bm_delta,bv_delta;
    public double[][] weight,w_delta,wm_delta,wv_delta;



    public Hidden(int inputsize, int h_count,double dropoutsize,double leaky,double learn_rate,int mini_batch){
        this.weight = new double[h_count][inputsize];
        this.w_delta = new double[h_count][inputsize];
        this.wm_delta = new double[h_count][inputsize];
        this.wv_delta = new double[h_count][inputsize];

        this.bias = new double[h_count];
        this.b_delta = new double[h_count];
        this.bm_delta = new double[h_count];
        this.bv_delta = new double[h_count];

        this.input_size = inputsize; this.h_count = h_count;

        this.dropoutsize = dropoutsize; this.leaky = leaky;
        this.learn_rate = learn_rate;

        this.mini_batch = mini_batch;

        this.renew_dropout(dropoutsize);
        // Initialize weight
        for (int i = 0; i < h_count; i++) {
//            this.bias[i] = Lab3.randomNormal(0.001) ;
            for (int j = 0; j < inputsize; j++) {
                this.weight[i][j] = (Tool.random()-0.5)/ Math.sqrt((double)this.input_size);
            }
        }

    }

    /**
     * Used for renew the dropout array after each epoch
     * @param dropsize
     *
     * @return return the new dropout array
     */

    public void renew_dropout(double dropsize){
        this.dropout = new double[this.h_count];
        for (int i = 0; i < this.h_count; i++) {
            dropout[i] = (Math.random() >= dropsize)? 1:0.0;
        }
    }

    /**
     * @param input
     *            input array foe this hidden layer
     * @return output of this layer, without activation function;
     */
    public double[] compute(double[] input){
        if(input.length != this.input_size){
            System.err.println("Wrong input size for Hidden Layer");
            System.exit(1);
        }
        double[] result = new double[h_count];
        for (int i = 0; i < h_count; i++) {
            result[i] = bias[i]* dropout[i];
            for (int j = 0; j < input_size; j++) {
                result[i] += this.weight[i][j] * input[j]* dropout[i];
            }
        }
        return lrelu_out(result);
    }


    /**
     * @param h_out
     *            output of this layer, without activation function;
     * @return output of this layer, with leaky relu;
     */

    public double[] lrelu_out(double[] h_out){
        double[] output = new double[this.h_count];
        for (int i = 0; i < this.h_count; i++)
            output[i] = (h_out[i] > 0)? h_out[i]:Tool.leaky_relu(h_out[i],this.leaky);
        return output;
    }


    /**
     * Used to update the weights of this layer
     * @param part1
     *              gradient backprop from other layers
     * @param out
     *                output of this layer
     * @param input
     *                input of this hidden layer
     *
     */
    public void update(double[] part1, double[] out,double[] input){
        // calculate the gradient for weights and bias

        double[][] w_gradient = new double[this.h_count][this.input_size];
        double[] b_gradient = new double[this.h_count];

        double lambda;
        for (int i = 0; i < this.h_count; i++) {
            lambda = (out[i] > 0)? 1:leaky ;
            b_gradient[i] = part1[i] * lambda * dropout[i];

            for (int j = 0; j < this.input_size; j++) {
                w_gradient[i][j] =  part1[i]*input[j]*lambda * dropout[i] ;

            }
        }
        this.batch_w_gradient.add(w_gradient);
        this.batch_b_gradient.add(b_gradient);

        if (this.batch_w_gradient.size() == this.mini_batch){

            this.update_count += 1;

            double[][] w_mean_gradient = new double[this.h_count][this.input_size];
            double[] b_mean_gradient= new double[this.h_count];
            for (int i = 0; i < this.batch_w_gradient.size(); i++) {
                w_mean_gradient = Tool.mat_2d(w_mean_gradient,this.batch_w_gradient.get(i),1,1/(double)this.mini_batch,"add");
                b_mean_gradient = Tool.mat_1d(b_mean_gradient,this.batch_b_gradient.get(i),1,1/(double)this.mini_batch,"add");
            }
            this.batch_w_gradient = new ArrayList<>();
            this.batch_b_gradient = new ArrayList<>();

            //adam item for weight
            wm_delta = Tool.mat_2d(wm_delta,w_mean_gradient,beta_1,1 - beta_1 ,"add");
            wv_delta = Tool.mat_2d(wv_delta,Tool.mat_2d(w_mean_gradient,null,2,1,"constant_pow"),
                    beta_2,1-beta_2,"add");


            //adam item for bias
            bm_delta = Tool.mat_1d(bm_delta,b_mean_gradient,beta_1,1-beta_1,"add");
            bv_delta = Tool.mat_1d(bv_delta,Tool.mat_1d(b_mean_gradient,null,2,1,"constant_pow"),
                    beta_2,1-beta_2,"add");


            for (int i = 0; i < h_count; i++) {
                bias[i] = bias[i] - this.learn_rate/(Math.sqrt(bv_delta[i]/(1-Math.pow(beta_2,update_count))) + 10e-8)
                        *bm_delta[i];
                for (int j = 0; j < input_size; j++) {
                    weight[i][j] = weight[i][j] - this.learn_rate / (Math.sqrt(wv_delta[i][j]/(1-Math.pow(beta_2,update_count)))
                            + 10e-8) * wm_delta[i][j];
                }
            }

        }

    }

    /**
     * Back Propagation to previous layers
     * @param part1
     *              gradient backprop from other layers
     * @param out
     *                output of this layer
     * @param input
     *                input of this hidden layer
     * @return 3-d array filled gradient passed to previous layer
     */
    public double[] h_back(double[] part1, double[] out,double[] input){
        double[] res = new double[input.length];
        for (int i = 0; i < out.length; i++){
            if (dropout[i]==0) continue;
            double lambda = (out[i] > 0)? 1:leaky ;
            for (int j = 0; j < input.length; j++)
                res[j] += part1[i]*out[i]* weight[i][j]*lambda;
        }

        return res;
    }


}