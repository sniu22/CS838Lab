import java.util.ArrayList;

/**
 * Created by Lament on 2017/3/5.
 */
public class Out_Layer {
    public int input_size, mini_batch, update_count = 0;
    public double learn_rate, beta_1 = 0.9, beta_2 = 0.999;
    public double[][] weight, w_delta, wm_delta, wv_delta;

    public ArrayList<double[][]> batch_w_gradient = new ArrayList<>();
    public ArrayList<double[]> batch_b_gradient = new ArrayList<>();

    public double[] bias, b_delta, bm_delta, bv_delta;
    public static int output_size = 6;
    public String activation;


    public Out_Layer(int input_size, String act_type, double learn_rate, int mini_batch) {
        this.activation = act_type;
        this.input_size = input_size;
        this.learn_rate = learn_rate;
        this.bias = new double[output_size];
        this.b_delta = new double[output_size];
        this.bv_delta = new double[output_size];
        this.bm_delta = new double[output_size];

        this.weight = new double[output_size][input_size];
        this.w_delta = new double[output_size][input_size];
        this.wm_delta = new double[output_size][input_size];
        this.wv_delta = new double[output_size][input_size];

        this.mini_batch = mini_batch;

        for (int i = 0; i < output_size; i++) {
//            this.bias[i] = Lab3.randomNormal(0.001) ;
            for (int j = 0; j < input_size; j++) {
                this.weight[i][j] = (Tool.random()-0.5) / Math.sqrt(this.input_size);
            }
        }
    }

    /**
     * @param input input array foe this hidden layer
     * @return output of this layer, without activation function;
     */
    public double[] output(double[] input) {
        if (input.length != this.input_size) {
            System.err.println("Wrong input size for Out Layer");
            System.exit(1);
        }
        double sum = 0;
        double[] result = new double[output_size];
        for (int i = 0; i < output_size; i++) {
            result[i] = bias[i];
            for (int j = 0; j < input_size; j++) {
                result[i] += this.weight[i][j] * input[j];
            }

            switch (this.activation) {
                case "softmax":
                    result[i] = Math.exp(result[i]);
                    sum += result[i];
                    break;
                case "sigmoid":
                    result[i] = 1 / (1 + Math.exp(-result[i]));
                    break;
            }
        }

        switch (this.activation) {
            case "softmax":
                return Tool.mat_1d(result, null, 1 / sum, 1, "mul_contant");
            case "sigmoid":
                return result;
        }
        return result;
    }


    public void update(double[] out, double[] input, int label) {
        // get the part1
        double[] part1 = new double[output_size];
        for (int i = 0; i < output_size; i++) {
            double lambda = (label == i) ? 1 : 0;
            switch (activation) {
                case "softmax":
                    part1[i] = out[i] - lambda;
                    break;
                case "sigmoid":
                    part1[i] = (out[i] - lambda) * out[i] * (1 - out[i]);
                    break;
            }
        }

        // calculate the gradient
        double[][] w_gradient = new double[output_size][this.input_size];
        double[] b_gradient = new double[output_size];

        for (int i = 0; i < output_size; i++) {
            b_gradient[i] = part1[i];
            for (int j = 0; j < this.input_size; j++) {
                w_gradient[i][j] = part1[i] * input[j];
            }
        }

        this.batch_w_gradient.add(w_gradient);
        this.batch_b_gradient.add(b_gradient);
        if (this.batch_w_gradient.size() == this.mini_batch) {
            this.update_count += 1;

            double[][] w_mean_gradient = new double[output_size][this.input_size];
            double[] b_mean_gradient = new double[output_size];

            for (int i = 0; i < this.batch_w_gradient.size(); i++) {
                w_mean_gradient = Tool.mat_2d(w_mean_gradient, this.batch_w_gradient.get(i), 1, 1 / (double)this.mini_batch, "add");
                b_mean_gradient = Tool.mat_1d(b_mean_gradient, this.batch_b_gradient.get(i), 1, 1 / (double)this.mini_batch, "add");

            }
//            System.out.print("wm " + Tool.print_array(w_mean_gradient[0]));
//            System.out.print("part1 " + Tool.print_array(part1));



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


            for (int i = 0; i < output_size; i++) {
                bias[i] = bias[i] - this.learn_rate/(Math.sqrt(bv_delta[i]/(1-Math.pow(beta_2,update_count))) + 10e-8)
                        *bm_delta[i];
                for (int j = 0; j < input_size; j++) {
                    weight[i][j] = weight[i][j] - this.learn_rate/
                            (Math.sqrt(wv_delta[i][j]/(1-Math.pow(beta_2,update_count))) + 10e-8) *
                            wm_delta[i][j];
                }
            }
        }

    }


    public double[] f_back(double[] out, double[] input, int label) {

        double[] result = new double[input_size];
        double[] part1 = new double[output_size];

        for (int i = 0; i < output_size; i++) {
            double lambda = (label == i) ? 1 : 0;
            switch (activation) {
                case "softmax":
                    part1[i] = out[i] - lambda;
                    break;
                case "sigmoid":
                    part1[i] = (out[i] - lambda) * out[i] * (1 - out[i]);
                    break;
                default: System.out.println("Ouput Function Activation Error");
            }
        }

        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                result[j] += part1[i] * weight[i][j];
            }
        }
        return result;
    }
}















