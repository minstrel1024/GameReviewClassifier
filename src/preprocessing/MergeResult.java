package preprocessing;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;

public class MergeResult {
	public static HashMap<String, Comment> cmap;
	public static void main(String[] args) throws Exception{
		try {
			BufferedReader br;

			Comment cnow;
			String route = "data/backup/";
			String fileh = "result";
			String tagfile = "data/tag_comments.csv";
			String notagfile = "data/no_tag_comments.csv";
			String originfile = "data/origin_comments.csv";
			//String originfile = "data/origin_comments.csv";
			String foreignfile = "data/foreign_comments.csv";
			String finalfile = "final.data";
			
			String trainfile = "data/train_comments.csv";
			String testfile = "data/test_comments.csv";
			String evalfile = "data/eval_comments.csv";
			
			int startcount = 1, endcount = 7;
			cmap = new HashMap<String, Comment>();
			for(int i=startcount; i<=endcount; i++){
				br = new BufferedReader(new FileReader(route+fileh+i+".csv"));
				String s;
				while((s = br.readLine())!=null){
					cnow = Comment.ParseString(s);

					//cnow.comment = cnow.comment.replaceAll("\"", "'");
					if(cnow != null){
						if(!cmap.containsKey(cnow.href))
							cmap.put(cnow.href, cnow);
					}
				}
				br.close();
			}
			BufferedWriter bw1 = new BufferedWriter(new FileWriter(tagfile));
			BufferedWriter bw2 = new BufferedWriter(new FileWriter(notagfile));
			BufferedWriter bw3 = new BufferedWriter(new FileWriter(foreignfile));
			BufferedWriter bwo = new BufferedWriter(new FileWriter(originfile));
			BufferedWriter bwf = new BufferedWriter(new FileWriter(finalfile));
			int cc = 0, cp = 0, cn = 0, cf = 0, ctrain = 0, ctest = 0;
			for(Comment c: cmap.values()){
				//boolean isEnglish = c.comment.matches("^[A-Za-z0-9\\.\\,\\?\\:\\!\\;/\\=\\%\\-\\&'\\(\\)\\$\\t\\+\\*\\^\\@\\_\\[\\]\\{\\}\\~\\| ]+$");
				boolean isEnglish = c.comment.matches("^[\\w\\pP\\p{Punct} ]+$");
				if(isEnglish){
					cc++;
					bw1.write(""+cc+","+c.tag+","+c.toString()+"\n");
					bw2.write(""+cc+","+c.toString()+"\n");
					bwo.write(""+cc+","+c.toOriginString()+"\n");
					String nlabel;
					if(c.tag == 1)
						nlabel = "positive";
					else
						nlabel = "negative";
					bwf.write(c.comment+"\t"+nlabel+"\n");
					if(c.tag == 1)
						cp ++;
					if(c.tag == -1)
						cn ++;
				}else{
					cf ++;
					if(cf < 100)
						System.out.println(c.comment+"\n");
					bw3.write(""+cf+","+c.tag+","+c.toOriginString()+"\n");
				}
			}
			bw1.close();
			bw2.close();
			bw3.close();
			bwo.close();
			bwf.close();
			System.out.println("Total English Comments: "+ cc);
			System.out.println("Positive Comments: "+ cp);
			System.out.println("Negative Comments: "+ cn);
			System.out.println("Foreign Comments: "+ cf);
			
			BufferedWriter bwtrain = new BufferedWriter(new FileWriter(trainfile));
			BufferedWriter bwtest = new BufferedWriter(new FileWriter(testfile));
			BufferedWriter bweval = new BufferedWriter(new FileWriter(evalfile));
			ctrain = (int) (cc*0.9);
			ctest = cc - ctrain;
			cc = 0;
			for(Comment c: cmap.values()){
				boolean isEnglish = c.comment.matches("^[\\w\\pP\\p{Punct} ]+$");
				if(isEnglish){
					cc++;
					if(cc <= ctrain)
						bwtrain.write(""+cc+","+c.tag+","+c.toString()+"\n");
					else {
						bwtest.write(""+cc+","+c.toString()+"\n");
						bweval.write(""+cc+","+c.tag+","+c.toString()+"\n");
					}
				}
			}
			bwtrain.close();
			bwtest.close();
			bweval.close();
			System.out.println("Train Comments: "+ ctrain);
			System.out.println("Test Comments: "+ ctest);

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}


