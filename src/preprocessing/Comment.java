package preprocessing;

public class Comment {
	public String href, uid, appid, comment, origin;
	int tag;
	Comment(){
		
	}
	public static Comment ParseString(String s){
		Comment c = new Comment();
		String ss[] = s.split(",", 3);
		if(ss.length < 3)
			return null;
		c.href = ss[0].trim();
		c.tag = Integer.parseInt(ss[1]);
		c.origin = RawComment(ss[2]);
		c.comment = ProcessComment(ss[2]);
		String[] cs = ss[0].split("/");
		if(cs.length < 7)
			return null;
		c.uid = cs[4];
		c.appid = cs[6];
		return c;
		
	}
	public static String RawComment(String s){
		s = s.trim();
		//s = s.replaceAll("[¡¯`]", "'");
		s = s.replace("&lt;", "<");
		s = s.replace("&gt;", ">");
		s = s.replace("&quot;", "\" ");
		s = s.replace("&amp;", "&");
		//s = s.replaceAll("\\s{1,}", " ");
		return s;
	}
	public static String ProcessComment(String s){
		s = s.trim();
		s = s.replaceAll("[¡¯`]", "'");
		s = s.replace("&lt;", "<");
		s = s.replace("&gt;", ">");
		s = s.replace("&quot;", "\" ");
		s = s.replace("&amp;", "&");
		s = s.replaceAll("\\?{2,}", " {?} ");
		StringBuffer sb = new StringBuffer();
		int l = s.length();
		boolean ispunc = false;
		boolean isurl = false;
		for(int i=0;i<l;i++){
			char c = s.charAt(i);
			String cs = new String(""+c);
			if(cs.matches("^[\\s]$")){
				ispunc = false;
				isurl = false;
				sb.append(cs);
			}else if((c == ':' && i<l-2 && s.charAt(i+1) == '/' && s.charAt(i+2)=='/')){
				sb.append(cs);
				ispunc = true;
				isurl = true;
			}else if(cs.matches("^[\\pP\\p{Punct}]$") && !(c == '/' && ((i!=0 && s.charAt(i-1)>='0' && s.charAt(i-1)<='9' )|| isurl)) && !((c == '.'||c == '%') && i!=0 && s.charAt(i-1)>='0' && s.charAt(i-1)<='9') && !(c == '-')){
				if(!ispunc && i!=0)
					sb.append(" "+cs);
				else
					sb.append(cs);
				ispunc = true;
				if(cs.matches("^[/]$")){
					sb.append(" ");
					ispunc = false;
				}
			}else{
				sb.append(cs);
			}
		}
		s = sb.toString();
		s = s.replaceAll("\\s{1,}", " ");
		String[] ss = s.split(" ");
		sb = new StringBuffer();
		l = ss.length;
		for(int i=0;i<l;i++){
			if(i!=0)
				sb.append(" ");
			if(ss[i].startsWith("http://")||ss[i].startsWith("https://"))
				sb.append("<url>");
			else
				sb.append(ss[i]);
		}
		s = sb.toString();
		return s;
	}
	public String toString(){
		return href+","+uid+","+appid+","+comment;
	}
	public String toOriginString(){
		return href+","+uid+","+appid+","+origin;
	}
}
