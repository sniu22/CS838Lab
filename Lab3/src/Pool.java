/**
 * Created by Lament on 2017/3/2.
 */
public class Pool {

    public int size;
    public int channel;
    public int in_width;
    public int out_width;

    public Pool(int size,int channel,int width){
        this.size = size;
        this.channel = channel;
        this.in_width = width;
        this.out_width = this.in_width/this.size;
    }

    /**
     * @param data
     *            The data received from conv layer;
     * @return The result of max pooling;
     */
    public double[][][] pool(double[][][] data){

        double[][][] out = new double[channel][out_width][out_width];
        for (int outc = 0; outc < channel; outc++)
            for (int i = 0; i < in_width; i++)
                for (int j = 0; j < in_width ; j++)
                    out[outc][i/size][j/size] =  Math.max(out[outc][i/size][j/size],data[outc][i][j]);
        return out;
    }

    /**
     * @param data
     *            The data received from conv layer;
     * @return One dimension array passed to hidden layer;
     */
    public double[] to_hidden(double[][][] data){ return Tool.to_line(this.pool(data));}

    /**
     * @param output
     *            outout result from this pooling layer
     * @return 3-d array , resized to ths input size of this pooling layer
     */
    public double[][][] depool(double[][][] output){
        double[][][] res = new double[channel][in_width][in_width];

        for (int i = 0; i < channel; i++)
            for (int j = 0; j < in_width; j++)
                for (int k = 0; k < in_width; k++)
                    res[i][j][k] = output[i][j/size][k/size];

        return res;
    }

    /**
     * @param conv_out
     *                output of conv layer
     * @param pool_out
     *                output of this pool layer
     *
     * @return 3-d array filled with 0 or 1 indicating the value of used pool_out
     */
    public double[][][] pool_signal(double[][][] conv_out, double[][][] pool_out){
        double[][][] out = new double[channel][in_width][in_width];
        for (int outc = 0; outc < channel; outc++)
            for (int i = 0; i < in_width; i++)
                for (int j = 0; j < in_width ; j++)
                    out[outc][i][j] = (pool_out[outc][i/size][j/size] == conv_out[outc][i][j])? 1:0;
        return out;
    }


    /**
     * @param part1
     *              gradient backprop from other layers(the layer after pool is conv).
     * @param conv_out
     *                output of conv layer
     * @param pool_out
     *                output of this pool layer
     *
     * @return 3-d array filled gradient passed to previous conv layer
     */

    public double[][][] cp_back(double[][][] part1, double[][][] conv_out, double[][][] pool_out){
        double[][][] repart1 = depool(part1);
        double[][][] signal = this.pool_signal(conv_out,pool_out);
        double[][][] res = Tool.mat_3d(repart1,signal,1,1,"mul");
        return res;
    }

    /**
     * @param part1
     *              gradient backprop from other layers(the layer after pool is hidden).
     * @param conv_out
     *                output of conv layer
     * @param pool_out
     *                output of this pool layer
     *
     * @return 3-d array filled with 0 or 1 indicating the value of used pool_out
     */

    public double[][][] hp_back(double[] part1, double[][][] conv_out, double[][][] pool_out){
        double[][][] repart1 = depool(Tool.to_matrix(part1,channel,out_width,out_width));
        double[][][] signal = this.pool_signal(conv_out,pool_out);
        double[][][] res = Tool.matrix_mul_3d(repart1,signal);
        return res;
    }


}
