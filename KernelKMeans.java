/**
 * Created by Eliyahu Kiperwasser on 13/05/2014.
 * Simple command line clustering tool using a variant of kernel k-kmeans.
 */

import java.io.*;
import java.util.*;

public class KernelKMeans {
    public static double dot(double []a, double []b, double c)
    {
        int dims = Math.min(a.length, b.length);
        double result = 0.0;

        for( int i = 0 ; i < dims ; i++)
            result += Math.pow(a[i] - b[i], 2);

        return 1.0 - (result / (result + c));
    }

    public static HashMap<String, String> parseFlags(String []args)
    {
        HashMap<String, String> flags = new HashMap<String, String>();
        for(int i = 0 ; i < args.length; i += 2)
            flags.put(args[i].toLowerCase(), args[i+1]);
        return flags;
    }

    public static HashMap<String, double[]> parseData(BufferedReader reader) throws IOException
    {
        HashMap<String, double[]> data = new HashMap<String, double[]>();

        String line;

        while((line = reader.readLine()) != null)
        {
            String []parts = line.split(" ");
            double []vec = new double[parts.length - 1];

            for(int i = 1 ; i < parts.length ; i++)
                vec[i-1] = Double.parseDouble(parts[i]);

            data.put(parts[0], vec);
        }

        return data;
    }

    public static HashMap<String, Integer> initLabels(Collection<String> keys, int nClusters)
    {
        HashMap<String, Integer> labels = new HashMap<String, Integer>();
        for(String key : keys)
            labels.put(key, Math.abs(key.hashCode()) % nClusters);

        return labels;
    }

    public static boolean improveLabels(HashMap<String, Integer> labels, HashMap<String, double[]> distances, int nClusters)
    {
        double []norm = new double[nClusters];

        for (double[] dists : distances.values())
            for(int iCluster = 0 ; iCluster < nClusters ; iCluster++)
                norm[iCluster] += dists[iCluster];

        double err = 0.0;
        boolean stop = true;

        for (Map.Entry<String, double[]> distEntry : distances.entrySet()) {
            int argMin = 0;
            double[] dists = distEntry.getValue();

            for (int iCluster = 1 ; iCluster < nClusters ; iCluster++)
                if ( dists[argMin] / (norm[argMin] + 0.0001) > dists[iCluster] / (norm[iCluster] + 0.0001) )
                    argMin = iCluster;

            if(argMin != labels.get(distEntry.getKey()))
                stop = false;
            labels.put(distEntry.getKey(), argMin);
            err += dists[argMin];
        }

        System.out.println("Current error " + err);

        return stop;
    }

    public static HashMap<String, double[]> calculateDistances(HashMap<String, double[]> data, HashMap<String, Integer> labels, double c, int nClusters)
    {
        HashMap<String, double[]> distances = new HashMap<String, double[]>();
        HashMap<String, double[]> dots = new HashMap<String, double[]>();
        int[] clusterHist = new int[nClusters];
        double[] interCluster = new double[nClusters];

        for (String key : data.keySet()) {
            distances.put(key, new double[nClusters]);
            dots.put(key, new double[nClusters]);
            clusterHist[labels.get(key)]++;
        }

        for (Map.Entry<String, double[]> firstWordEntry : data.entrySet()) {
            String wordA = firstWordEntry.getKey();
            double[] valA = firstWordEntry.getValue();
            int labelA = labels.get(wordA);

            for (Map.Entry<String, double[]> secondWordEntry : data.entrySet()) {
                double temp = dot(valA, secondWordEntry.getValue(), c);
                dots.get(secondWordEntry.getKey())[labelA] += temp;
                interCluster[labelA] += (labels.get(secondWordEntry.getKey()) == labelA ? temp : 0.0);
            }
        }

        for (int iCluster = 0; iCluster < nClusters; iCluster++)
            interCluster[iCluster] /= (clusterHist[iCluster] * clusterHist[iCluster]);

        for (Map.Entry<String, double[]> wordEntry : data.entrySet()) {
            String word = wordEntry.getKey();
            double[] val = wordEntry.getValue();
            double selfProd = dot(val, val, c);

            for (int iCluster = 0; iCluster < nClusters; iCluster++)
                distances.get(word)[iCluster] = selfProd + interCluster[iCluster]
                        - (2.0 * dots.get(word)[iCluster] / clusterHist[iCluster]);
        }

        return distances;
    }

    public static void main(String []args) throws IOException
    {
        if(args[0] == "-h" || args.length % 2 != 0)
        {
            System.out.println("Usage: KernelKMeans <-i input> <-n num of clusters> <-o output> <-c kernel parameter>");
            return;
        }

        HashMap<String, String> flags = parseFlags(args);
        int nClusters = Integer.parseInt(flags.get("-n"));
        double c = Double.parseDouble(flags.get("-c"));

        BufferedReader reader = new BufferedReader(flags.containsKey("-i") ? new FileReader(flags.get("-i")) : new InputStreamReader(System.in));

        HashMap<String, double[]> data = parseData(reader);
        HashMap<String, Integer> labels = initLabels(data.keySet(), nClusters);

        boolean stop = false;

        for(int iEpoch = 0 ; (!stop) ; iEpoch++) {
            System.out.println("Epoch " + iEpoch);
            HashMap<String, double[]> distances = calculateDistances(data, labels, c, nClusters);
            stop = improveLabels(labels, distances, nClusters);
        }

        PrintStream ps = flags.containsKey("-o") ? new PrintStream(new FileOutputStream(flags.get("-o"))) : System.out;

        for(Map.Entry<String, Integer> wordEntry : labels.entrySet())
            ps.println(wordEntry.getKey() + " " + wordEntry.getValue());

        ps.close();
    }
}
