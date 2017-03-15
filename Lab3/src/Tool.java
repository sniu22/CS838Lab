import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by Shuo Niu on 2017/3/11.
 */
public class Tool  {

    public static void main(String[] args){
        int a = 5,c=19,d = 4;
        double b = 0.5;

        System.out.println(b*a);
        System.out.println(d/a);

    }

    /**
     * Some activation functions
     */

    public static double leaky_relu(double input,double leaky){return (input > 0)? input:leaky*input ;}


    /**
     * Some matrix manipulation
     */

    public static double[] to_line(double[][][] data){
        double[] res = new double[data.length * data[0].length * data[0][0].length];

        int count = 0;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                for (int k = 0; k < data[0][0].length; k++) {
                    res[count] = data[i][j][k];
                    count++;
                }
            }
        }

        return res;
    }

    public static double[][][] to_matrix(double[] data, int channel,int width, int height ){
        double[][][] res = new double[channel][width][height];
        int count = 0;
        for (int i = 0; i < channel; i++) {
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < height; k++) {
                    res[i][j][k] = data[count];
                    count ++;
                }
            }
        }
        return res;
    }


    /**
     * Some matrix computation functions
     */

    public static double[][][] matrix_mul_3d(double[][][] ma, double[][][] mb){
        int channel = ma.length;
        int width = ma[0].length;
        int height= ma[0].length;

        double[][][] out = new double[channel][width][height];
        for (int i = 0; i < channel; i++)
            for (int j = 0; j < width; j++)
                for (int k = 0; k < height; k++)
                    out[i][j][k] = ma[i][j][k] * mb[i][j][k];
        return out;
    }


    public static double[] mat_1d(double[] a, double[] b, double c1, double c2, String action) {
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            switch (action) {
                case "add_constant":
                    res[i] = a[i] + c1; break;
                case "mul_contant":
                    res[i] = a[i] * c1; break;
                case "constant_div":
                    res[i] = c1 / a[i]; break;
                case "constant_pow":
                    res[i] = Math.pow(a[i],c1); break;
                case "add":
                    res[i] = c1 * a[i] + c2 * b[i]; break;
                case "mul":
                    res[i] = c1 * a[i] * b[i]; break;
                case "divide":
                    res[i] = c1 * a[i] / b[i]; break;
                default: System.out.println("Mat 1d Undefined action");
            }
        }
        return res;
    }

    public static double[][] mat_2d(double[][] a, double[][] b, double c1, double c2, String action) {
        double[][] res = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                switch (action) {
                    case "add_constant":
                        res[i][j] = a[i][j] + c1;
                        break;
                    case "mul_contant":
                        res[i][j] = a[i][j] * c1;
                        break;
                    case "constant_div":
                        res[i][j] = c1 / a[i][j];
                        break;
                    case "constant_pow":
                        res[i][j] = Math.pow(a[i][j], c1);
                        break;
                    case "add":
                        res[i][j] = c1 * a[i][j] + c2 * b[i][j];
                        break;
                    case "mul":
                        res[i][j] = c1 * a[i][j] * b[i][j];
                        break;
                    case "divide":
                        res[i][j] = c1 * a[i][j] / b[i][j];
                        break;
                    default:
                        System.out.println("Mat 2d Undefined action");
                }
            }
        }
        return res;
    }

    public static double[][][] mat_3d(double[][][] a, double[][][] b, double c1 ,double c2 , String action){
        double[][][] res = new double[a.length][a[0].length][a[0][0].length];
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < a[0].length; j++)
                for (int k = 0; k < a[0][0].length; k++)
                    {
                        switch (action){
                            case "add_constant":
                                res[i][j][k] = a[i][j][k] + c1; break;
                            case "mul_contant":
                                res[i][j][k] = a[i][j][k] * c1; break;
                            case "constant_div":
                                res[i][j][k] = c1 / a[i][j][k]; break;
                            case "add":
                                res[i][j][k] = c1 * a[i][j][k] + c2*b[i][j][k] ;break;
                            case "mul":
                                res[i][j][k] = c1 * a[i][j][k] * b[i][j][k] ;break;
                            case "divide":
                                res[i][j][k] = c1 * a[i][j][k] / b[i][j][k] ;break;
                            default: System.out.println("Mat 3d Undefined action");
                        }
                    }
        return res;
    }

    public static double[][][][] mat_4d(double[][][][] a, double[][][][] b, double c1 ,double c2 , String action){
        double[][][][] res = new double[a.length][a[0].length][a[0][0].length][a[0][0][0].length];
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < a[0].length; j++)
                for (int k = 0; k < a[0][0].length; k++)
                    for (int l = 0; l < a[0][0][0].length; l++){
                        switch (action){
                            case "add_constant":
                                res[i][j][k][l] = a[i][j][k][l] + c1; break;
                            case "mul_constant":
                                res[i][j][k][l] = a[i][j][k][l] * c1; break;
                            case "constant_div":
                                res[i][j][k][l] = c1 / (a[i][j][k][l] + c2); break;
                            case "constant_pow":
                                res[i][j][k][l] = Math.pow(a[i][j][k][l],c1); break;
                            case "add":
                                res[i][j][k][l] = c1 * a[i][j][k][l] + c2*b[i][j][k][l] ;break;
                            case "mul":
                                res[i][j][k][l] = c1 * a[i][j][k][l] * b[i][j][k][l] ;break;
                            case "sqrt":
                                res[i][j][k][l] = Math.sqrt(a[i][j][k][l]) ;break;
                            case "divide":
                                res[i][j][k][l] = c1 * a[i][j][k][l] / b[i][j][k][l] ;break;
                            default: System.out.println("Mat_4d: Wrong String Action"); System.exit(0);
                        }
                    }
        return res;
    }





    public static String print_array(double[] input){
        String  res= "  ";

        for (int i = 0; i < input.length; i++)
            res += Double.toString(input[i]) + " ; ";

        res += "\n";
        return res;
    }

    public static String print_2darray(double[][] input){
        String  res= "  ";

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                res += input[i][j] + " , ";
            }
        }
        res += "\n";
        return res;
    }

    public static String print_3darray(double[][][] input){
        String  res= "";
        for (int c = 0; c < input.length; c++) {
            res += "Channel" + c +"\n";
            for (int i = 0; i < input[0].length; i++) {
                for (int j = 0; j < input[0][0].length; j++) {
                    res += input[c][i][j] + "  ";
                }
                res += "\n";
            }
            res += "\n";
        }
        return res;
    }



    /**
     * Some image processing functions
     */
    public static double[][] image_normalization(int[][] channel){
        double[][] res = new double[channel.length][channel[0].length];
        double sum = 0;

        for (int i = 0; i < channel.length; i++) {
            for (int j = 0; j < channel[i].length; j++) {
                sum += channel[i][j];
            }
        }
        double mean = sum / channel.length / channel[0].length;

        for (int i = 0; i < channel.length; i++) {
            for (int j = 0; j < channel[i].length; j++) {
                res[i][j] = ((double)channel[i][j] - mean);
            }
        }
        return res;
    }

    public static double[][] get_channel(int[][] channel){
        double[][] res = new double[channel.length][channel[0].length];
        for (int i = 0; i < channel.length; i++) {
            for (int j = 0; j < channel[0].length; j++) {
                res[i][j] = (double) channel[i][j] / 255.0;
            }
        }

        return res;
    }

    public static double[][][] get_image(Instance input){
        double[][][] res = new double[4][input.getWidth()][input.getHeight()];
        res[0] = Tool.image_normalization(input.getRedChannel());
        res[1] = Tool.image_normalization(input.getGreenChannel());
        res[2] = Tool.image_normalization(input.getBlueChannel());
        res[3] = Tool.image_normalization(input.getGrayImage());
        return res;
    }

    public static double[][][] get_image2(Instance input){
        double[][][] res = new double[4][input.getWidth()][input.getHeight()];
        res[0] = Tool.get_channel(input.getRedChannel()).clone();
        res[1] = Tool.get_channel(input.getGreenChannel()).clone();
        res[2] = Tool.get_channel(input.getBlueChannel()).clone();
        res[3] = Tool.get_channel(input.getGrayImage()).clone();
        return res;
    }


    /**
     * Some random generating function
     */

    public static Random randomInstance = new Random(638);

    /**
     * @return The next random double.
     */
    public static double random() {
        return randomInstance.nextDouble();
    }

    /**
     * @param lower
     *            The lower end of the interval.
     * @param upper
     *            The upper end of the interval. It is not possible for the
     *            returned random number to equal this number.
     * @return Returns a random integer in the given interval [lower, upper).
     */
    public static int randomInInterval(int lower, int upper) {
        return lower + (int) Math.floor(random() * (upper - lower));
    }


    /**
     * @param upper
     *            The upper bound on the interval.
     * @return A random number in the interval [0, upper).
     * see Utils#randomInInterval(int, int)
     */
    public static int random0toNminus1(int upper) {
        return randomInInterval(0, upper);
    }

    public static double randomNormal(double variance) {
        return randomInstance.nextGaussian() * variance;
    }


    /**
     * Some instance handeling functions
     */

    private static int imageSize = 32;
    private static enum Category { airplanes, butterfly, flower, grand_piano, starfish, watch}

    protected static final double shiftProbNumerator = 6.0; // 6.0 is the 'default.'
    protected static final double probOfKeepingShiftedTrainsetImage = (shiftProbNumerator / 48.0);
    protected static final boolean perturbPerturbedImages = false;


    public static void loadDataset(Dataset dataset, File dir) {
        for (File file : dir.listFiles()) {
            // check all files
            if (!file.isFile() || !file.getName().endsWith(".jpg")) {
                continue;
            }
            //String path = file.getAbsolutePath();
            BufferedImage img = null, scaledBI = null;
            try {
                // load in all images
                img = ImageIO.read(file);
                // every image's name is in such format:
                // label_image_XXXX(4 digits) though this code could handle more than 4 digits.
                String name = file.getName();
                int locationOfUnderscoreImage = name.indexOf("_image");

                // Resize the image if requested.  Any resizing allowed, but should really be one of 8x8, 16x16, 32x32, or 64x64 (original data is 128x128).
                if (imageSize != 128) {
                    scaledBI = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
                    Graphics2D g = scaledBI.createGraphics();
                    g.drawImage(img, 0, 0, imageSize, imageSize, null);
                    g.dispose();
                }
                Instance instance = new Instance(scaledBI == null ? img : scaledBI, name, name.substring(0, locationOfUnderscoreImage));
                dataset.add(instance);
            } catch (IOException e) {
                System.err.println("Error: cannot load in the image file");
                System.exit(1);
            }
        }

    }

}

